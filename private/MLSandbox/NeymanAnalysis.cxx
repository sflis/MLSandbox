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

//double up_lim(std::vector<double> ts_distr, double cl){
//    return ts_distr[ts_distr[current_xi].size() * cl];
//}


struct NeymanThreadData{
    NeymanThreadData(boost::shared_ptr<NeymanAnalysis> ana,
                     FCRanks &globalRanks,
                     FCRanks &globalBestFits,
                     std::queue<double> &testHypothesisSet,
                     uint64_t nExperiments,
                     uint64_t threadNumber,
                     double cl,
                     uint64_t nHypotheses,
                     uint64_t &nTestedHypotheses
                    ):
                     ana(ana),
                     globalRanks(globalRanks),
                     globalBestFits(globalBestFits),
                     testHypothesisSet(testHypothesisSet),
                     nExperiments(nExperiments),
                     threadNumber(threadNumber),
                     cl(cl),
                     totalHypotheses(nHypotheses),
                     nTestedHypotheses(nTestedHypotheses)
                     {}

        boost::shared_ptr<NeymanAnalysis> ana;
        FCRanks &globalRanks;
        FCRanks &globalBestFits;
        std::queue<double> &testHypothesisSet;
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
/*
double NeymanAnalysis::ComputeLimit(double ts, double cl, double prec){



    bool found_lim = false;
    double current_xi = 0;
    tsDistributions_[current_xi] =  TestStatisticDistribution(0,100);
    double current_ts_up = up_lim(tsDistributions_[current_xi],cl);
    found_lim = ts>current_ts_up?true:false;
    //if limit is found when xi=0
    if(found_lim)
        return 0.0/0.0;

    do{
        current_xi += 10 * prec;
        tsDistributions_[current_xi] =  TestStatisticDistribution(0,100);
        current_ts_up = up_lim(tsDistributions_[current_xi], cl);
    }while(ts>current_ts_up)

    current_xi += 100 * prec;
    tsDistributions_[current_xi] =  TestStatisticDistribution(0,100);
    current_ts_up = up_lim(tsDistributions_[current_xi], cl);

    uint64_t bins = current_xi/prec+0.5;
    Histogram1D(bins, 0,current_xi);


    while(!found_lim){
        tsDistributions_


    }


}
*/

//_____________________________________________________________________________
void NeymanAnalysis::ComputeRanks(uint64_t nExperiments,
                double minXi,
                double maxXi,
                uint64_t nSteps,
                uint64_t nThreads){
    //initializing thread variables
    vector<pthread_t> threadHandels;
    vector<NeymanThreadData*> threadData;
    vector<int> threadId;
    threadHandels.resize(nThreads);

    //this is where the calculated best fit distributions are stored. NOTE:not implemented yet
    FCRanks globalBestFits;

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
    //Initiating the threads
    for(uint32_t i = 0; i < nThreads; i++){
        threadData.push_back(
            new NeymanThreadData(
                boost::shared_ptr<NeymanAnalysis>(
                    new NeymanAnalysis(*this,i)// boost::shared_ptr<Likelihood>( llh_->Clone(i) ))
                ),
                tsDistributions_,
                globalBestFits,
                llh_hypos,
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


    //Fetch first mu value to be computed
    //===Lock Fetch Hypothesis===++++++++++++++++++++++++
    pthread_mutex_lock( &mutexNeymanFetchHypothesis );
    cout<<"Starting Thread:"<<setw(2)<<neymanThreadData->threadNumber<<endl;
    bool muQueueEmpty = neymanThreadData->testHypothesisSet.empty();
    if(!muQueueEmpty){
        currentHypothesis = neymanThreadData->testHypothesisSet.front();
        neymanThreadData->testHypothesisSet.pop();
    }
    pthread_mutex_unlock( &mutexNeymanFetchHypothesis );
    //===UnLock Fetch Hypothesis===----------------------


    uint64_t n = neymanThreadData->nExperiments;
    ranks.resize(n);
    double cl = 0.9;
    while(!muQueueEmpty){

        for(uint64_t i = 0; i < n; i++){
            analysis.Sample(currentHypothesis);
            ranks[i] = analysis.EvaluateTestStatistic(currentHypothesis);
        }
        std::sort(ranks.begin(), ranks.end());
        currentRankAtCL = ranks[(1-cl)*(n-1)];
        //===Lock Write Ranks===++++++++++++++++
        pthread_mutex_lock( &mutexNeymanWriteRanks );
        neymanThreadData->nTestedHypotheses++;
        cout<<"=====Thread:"<<setw(2)<<neymanThreadData->threadNumber<<"====="<<endl;
        cout<<"Signal :"<<currentHypothesis<<endl;
        cout<<"Rank (@"<<cl*100<<"%CL and "<<n<<" experiments): "<<currentRankAtCL<<endl;
        cout<<"Hypotheses tested "<<neymanThreadData->nTestedHypotheses<<"/"<<neymanThreadData->totalHypotheses<<endl;
        cout<<"==================="<<endl;

        //Transfering the computed ranks to the global ranks object in the main thread
        neymanThreadData->globalRanks.Fill(currentHypothesis,ranks,false);

        pthread_mutex_unlock( &mutexNeymanWriteRanks );
        //===UnLock Write Ranks===--------------

        //===Lock Fetch Hypothesis===+++++++++++++++++++
        pthread_mutex_lock( &mutexNeymanFetchHypothesis );

        muQueueEmpty = neymanThreadData->testHypothesisSet.empty();
        if(!muQueueEmpty){
            currentHypothesis = neymanThreadData->testHypothesisSet.front();
            neymanThreadData->testHypothesisSet.pop();
        }

        pthread_mutex_unlock( &mutexNeymanFetchHypothesis );
        //===UnLock Fetch Hypothesis===-----------------

    }


    return 0;
}
