#include "MLSandbox/NeymanAnalysis.h"
#include "MLSandbox/FCRanks.h"

#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>

#include <pthread.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>


#include <boost/make_shared.hpp>
using namespace std;

struct job;
struct NeymanThreadData{
    NeymanThreadData(boost::shared_ptr<NeymanAnalysis> ana,
                     FCRanks &globalRanks,
                     FCRanks &globalBestFits,
                     std::queue<job> &jobQueue,//std::queue<double> &testHypothesisSet,
                     uint64_t nExperiments,
                     uint64_t threadNumber,
                     double cl,
                     uint64_t nHypotheses,
                     uint64_t &nTestedHypotheses
                    ):
                     ana(ana),
                     globalRanks(globalRanks),
                     globalBestFits(globalBestFits),
                     jobQueue(jobQueue),
                     nExperiments(nExperiments),
                     threadNumber(threadNumber),
                     cl(cl),
                     totalHypotheses(nHypotheses),
                     nTestedHypotheses(nTestedHypotheses)
                     {}

        boost::shared_ptr<NeymanAnalysis> ana;
        FCRanks &globalRanks;
        FCRanks &globalBestFits;
        std::queue<job> &jobQueue;
        uint64_t nExperiments;
        uint64_t threadNumber;
        double cl;
        uint64_t totalHypotheses;
        uint64_t &nTestedHypotheses;
};

void *tsComputationThread(void *data);

pthread_mutex_t mutexNeymanFetchHypothesis = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutexNeymanWriteRanks = PTHREAD_MUTEX_INITIALIZER;

NeymanAnalysis::NeymanAnalysis(NeymanAnalysis &analysis, int64_t seed): 
            computedBestFit_(false)
          , ranksCompSet_(false)
          , minimizer_(analysis.minimizer_)
{
    llh_ = boost::shared_ptr<Likelihood>(analysis.llh_->Clone(seed));

}

std::vector<double> NeymanAnalysis::TestStatisticDistribution(double xi, uint64_t n){

    std::vector<double> ts(n);
    //sample and evaluate teststatistic n times
    for(uint64_t i = 0; i<n; i++){
        Sample(xi);
        ts[i] = EvaluateTestStatistic(0);
    }

    std::sort(ts.begin(), ts.end());
    return ts;
}

double NeymanAnalysis::ComputeLimit(double ts, double cl, double prec){
    std::map<double, std::vector<double> > &tsDistrMap = tsDistributions_.ranks_;
    double up_lim = 0;
    for (const auto & tsDistr: tsDistrMap) {
        double tsAtCL = tsDistr.second[tsDistr.second.size() * (1-cl)];
        if(ts<tsAtCL){
            up_lim = tsDistr.first;
            break;
        }
    }
    return up_lim;
}


struct job
{
    job(double hypo =0, uint64_t nExperiments= 0,bool last=false):hypo(hypo),nExperiments(nExperiments),last(last){}

    double hypo;
    uint64_t nExperiments;
    bool last;
};

//_____________________________________________________________________________
void NeymanAnalysis::ComputeRanks(uint64_t nExperiments,
                double minXi,
                double maxXi,
                uint64_t nSteps,
                uint64_t nThreads,
                uint64_t maxExperimentsPerThread){
    //initializing thread variables
    vector<pthread_t> threadHandels;
    vector<NeymanThreadData*> threadData;
    vector<int> threadId;
    threadHandels.resize(nThreads);



    uint64_t globalnTestedHyptheses = 0;
    //There ought to be a better way to set up the xi values queue
    std::vector<double> llh_xis(nSteps);
    double stepSize= (maxXi-minXi)/nSteps;
    for(int64_t i = 0; i < nSteps; i++){
        llh_xis[i] = minXi+i*stepSize;
    }
    std::queue<double, std::deque<double> > llh_hypos(
        std::deque<double>(llh_xis.begin(), llh_xis.end())
    );

    std::queue<job> jobQueue;
    for(auto xi: llh_xis){
        int64_t N = nExperiments;

        while(N>maxExperimentsPerThread){
            jobQueue.push(job(xi,maxExperimentsPerThread,false));
            N -=maxExperimentsPerThread;
        }
        jobQueue.push(job(xi,maxExperimentsPerThread,true));
    }
    //Initiating the threads
    for(uint32_t i = 0; i < nThreads; i++){
        threadData.push_back(
            new NeymanThreadData(
                boost::shared_ptr<NeymanAnalysis>(
                    new NeymanAnalysis(*this,i)// boost::shared_ptr<Likelihood>( llh_->Clone(i) ))
                ),
                tsDistributions_,
                globalBestFits_,
                jobQueue,
                nExperiments,
                i,
                0.9,
                nSteps,
                globalnTestedHyptheses
            )
        );
        threadId.push_back(pthread_create(&threadHandels[i], NULL, tsComputationThread, (void*) threadData[i]) ) ;
    }

    //===Lock Fetch Mu===
    pthread_mutex_lock( &mutexNeymanFetchHypothesis );
    //variable that keeps track if the queue with mu vaules to compue is empty
    bool muQueueEmpty =llh_hypos.empty();
    pthread_mutex_unlock( &mutexNeymanFetchHypothesis );
    //===UnLock Fetch Mu===

    for(uint32_t i = 0; i < nThreads; i++){
        pthread_join( threadHandels[i], NULL);
    }
}

//_____________________________________________________________________________
void * tsComputationThread(void *data){
    //Grabbing the data in a convienient pointer
    NeymanThreadData *neymanThreadData = (NeymanThreadData *) data;
    NeymanAnalysis &analysis = *neymanThreadData->ana;


    vector<double> ranks;
    vector<double> muFits;
    double currentHypothesis = 0;
    double currentRankAtCL = 0;
    job currentJob;

    //Fetch first mu value to be computed
    //===Lock Fetch Hypothesis===++++++++++++++++++++++++
    pthread_mutex_lock( &mutexNeymanFetchHypothesis );
    cout<<"Starting Thread:"<<setw(2)<<neymanThreadData->threadNumber<<endl;
    bool muQueueEmpty = neymanThreadData->jobQueue.empty();
    if(!muQueueEmpty){
        currentJob = neymanThreadData->jobQueue.front();
        neymanThreadData->jobQueue.pop();
    }
    pthread_mutex_unlock( &mutexNeymanFetchHypothesis );
    //===UnLock Fetch Hypothesis===----------------------


    uint64_t n = currentJob.nExperiments;//neymanThreadData->nExperiments;
    ranks.resize(n);
    muFits.resize(n);

    double cl = 0.9;
    while(!muQueueEmpty){

        for(uint64_t i = 0; i < n; i++){
            analysis.Sample(currentJob.hypo);
            ranks[i] = analysis.EvaluateTestStatistic(currentJob.hypo);
            muFits[i] = analysis.minimizer_.bestFit_;

        }
        std::sort(ranks.begin(), ranks.end());
        currentRankAtCL = ranks[(1-cl)*(n-1)];
        if(currentJob.last){    

            //===Lock Write Ranks===++++++++++++++++
            pthread_mutex_lock( &mutexNeymanWriteRanks );
            neymanThreadData->nTestedHypotheses++;
            cout<<"=====Thread:"<<setw(2)<<neymanThreadData->threadNumber<<"====="<<endl;
            cout<<"Signal :"<<currentJob.hypo<<endl;
            cout<<"Hypotheses tested "<<neymanThreadData->nTestedHypotheses<<"/"<<neymanThreadData->totalHypotheses<<endl;
            cout<<"==================="<<endl;

            //Transfering the computed ranks to the global ranks object in the main thread
            neymanThreadData->globalRanks.Fill(currentJob.hypo,ranks,false);
            neymanThreadData->globalBestFits.Fill(currentJob.hypo,muFits,false);
            pthread_mutex_unlock( &mutexNeymanWriteRanks );
            //===UnLock Write Ranks===--------------
        }
        else{
            //===Lock Write Ranks===++++++++++++++++
            pthread_mutex_lock( &mutexNeymanWriteRanks );
            //neymanThreadData->nTestedHypotheses++;
            cout<<"=====Thread:"<<setw(2)<<neymanThreadData->threadNumber<<"====="<<endl;
            cout<<"Signal :"<<currentJob.hypo<<endl;
            cout<<"Computed :"<<n<<" trials"<<endl;
            cout<<"==================="<<endl;

            //Transfering the computed ranks to the global ranks object in the main thread
            neymanThreadData->globalRanks.Fill(currentJob.hypo,ranks,false);
            neymanThreadData->globalBestFits.Fill(currentJob.hypo,muFits,false);
            pthread_mutex_unlock( &mutexNeymanWriteRanks );
            //===UnLock Write Ranks===--------------  
        }   

        //===Lock Fetch Hypothesis===+++++++++++++++++++
        pthread_mutex_lock( &mutexNeymanFetchHypothesis );

        muQueueEmpty = neymanThreadData->jobQueue.empty();
        if(!muQueueEmpty){
            currentJob = neymanThreadData->jobQueue.front();
            neymanThreadData->jobQueue.pop();
        }

        pthread_mutex_unlock( &mutexNeymanFetchHypothesis );
        //===UnLock Fetch Hypothesis===-----------------

    }


    return 0;
}
