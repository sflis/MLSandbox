#include "MLSandbox/FeldmanCousins.h"

#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>

#include <pthread.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <queue>

#include <boost/make_shared.hpp>

struct FCThreadData{
        FCThreadData( boost::shared_ptr<FeldmanCousinsAnalysis> ana,
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

        boost::shared_ptr<FeldmanCousinsAnalysis> ana;
        FCRanks &globalRanks;
        FCRanks &globalBestFits;
        std::queue<double> &testHypothesisSet;
        uint64_t nExperiments;
        uint64_t threadNumber;
        double cl;
        uint64_t totalHypotheses;
        uint64_t &nTestedHypotheses;
};

void *rankComputationThread(void *data);

pthread_mutex_t mutexFetchHypothesis = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutexWriteRanks = PTHREAD_MUTEX_INITIALIZER;

using namespace std;
//_____________________________________________________________________________
void FeldmanCousinsAnalysis::ComputeLimits(double &upper, double &lower){

    int status;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 0;

    //We need an initial fit to define the start interval
    if(!computedBestFit_ || llh_->Changed()){
                minimizer_.ComputeBestFit(*llh_);
                computedBestFit_ = true;
    }
    //minimizer_.ComputeBestFit(*llh_);
    double x_lo = 0.0, x_hi = minimizer_.bestFit_;
    gsl_function llhrCross;
    llhrCross.function = &FeldmanCousinsAnalysis::likelihoodRatioCrossings;
    llhrCross.params = this;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);

    double currentR = EvaluateTestsStatistic(0);
    double currentRCritical = ranks_.rCB(0);
    uint64_t max_iter = 100, iter=0;
    //Finding lower limit
    //This is only done if the best fit mu>0 and current likelihood ratio is smaller than the corresponding critical
    //value (so it can cross the critical boundary).
    if(minimizer_.bestFit_!=0 && currentR < currentRCritical){
        gsl_root_fsolver_set (s, &llhrCross, x_lo, x_hi);
        do{
            iter++;
            status = gsl_root_fsolver_iterate (s);
            r = gsl_root_fsolver_root (s);
            x_lo = gsl_root_fsolver_x_lower (s);
            x_hi = gsl_root_fsolver_x_upper (s);
            status = gsl_root_test_interval (x_lo, x_hi,0, 1e-10);
                                             if(r<0)
                                                 break;
        }while ((status == GSL_CONTINUE && iter < max_iter));
    }

    if(r<0)
        lower = 0;
    else
        lower = r;//llh_->Xi2Mu(r);

    if(minimizer_.bestFit_!=0){
        x_lo = minimizer_.bestFit_, x_hi = 2*minimizer_.bestFit_ - r + 0.00001;
    }
    else{
        x_lo = 0, x_hi = 0.0001;
    }

    currentR = likelihoodRatioCrossings(x_hi, this);

    while( currentR > 0){
        x_lo = x_hi;
        x_hi+=x_hi*0.1;
        currentR = likelihoodRatioCrossings(x_hi,this);
    }
    gsl_root_fsolver_set (s, &llhrCross, x_lo, x_hi);
    iter=0;

   //Finding upper limit
    do{
        iter++;
        status = gsl_root_fsolver_iterate (s);
        r = gsl_root_fsolver_root (s);
        x_lo = gsl_root_fsolver_x_lower (s);
        x_hi = gsl_root_fsolver_x_upper (s);
        status = gsl_root_test_interval (x_lo, x_hi, 0, 1e-10);
    }while ((status == GSL_CONTINUE && iter < max_iter) );

    upper = r;// llh_->Xi2Mu(r);

    gsl_root_fsolver_free (s);

}
//_____________________________________________________________________________
double FeldmanCousinsAnalysis::likelihoodRatioCrossings(double xi, void *params){
    return ((FeldmanCousinsAnalysis*) params)->EvaluateTestsStatistic(xi) - ((FeldmanCousinsAnalysis*) params)->ranks_.rCB(xi);
}
//_____________________________________________________________________________
void FeldmanCousinsAnalysis::ComputeRanks(uint64_t nExperiments,
                double minXi,
                double maxXi,
                uint64_t nSteps,
                uint64_t nThreads){
    //initializing thread variables
    vector<pthread_t> threadHandels;
    vector<FCThreadData*> threadData;
    vector<int> threadId;
    threadHandels.resize(nThreads);

    //this is where the calculated best fit distributions are stored. NOTE:not implemented yet
    FCRanks globalBestFits;

    uint64_t globalnTestedHyptheses = 0;
    //There ought to be a better way to set up the xi values queue
    std::vector<double> llh_xis(nSteps);
    double stepSize= (maxXi-minXi)/nSteps;
    for(uint64_t i = 0; i < nSteps; i++){
        llh_xis[i] = i*stepSize;
    }
    std::queue<double, std::deque<double> > llh_hypos(
        std::deque<double>(llh_xis.begin(), llh_xis.end())
    );
    //Initiating the threads
    for(uint32_t i = 0; i < nThreads; i++){
        threadData.push_back(
            new FCThreadData(
                boost::shared_ptr<FeldmanCousinsAnalysis>(
                    new FeldmanCousinsAnalysis( boost::shared_ptr<Likelihood>( llh_->Clone(i) ), cl_)
                ),
                ranks_,
                globalBestFits,
                llh_hypos,
                nExperiments,
                i,
                cl_,
                nSteps,
                globalnTestedHyptheses
            )
        );
        threadId.push_back(pthread_create(&threadHandels[i], NULL, rankComputationThread, (void*) threadData[i]) ) ;
    }

    //===Lock Fetch Mu===
    pthread_mutex_lock( &mutexFetchHypothesis );
    //variable that keeps track if the queue with mu vaules to compue is empty
    bool muQueueEmpty =llh_hypos.empty();
    pthread_mutex_unlock( &mutexFetchHypothesis );
    //===UnLock Fetch Mu===


//     while(!muQueueEmpty){
//         //wait for 2 min before writing out current progress.
//         //FIXME: this is stupid for very fast runs as they will be at least 2 min long
//         //a better solution should be made.
//         sleep(120);
//
//         //===Lock Write r-crit===
//         pthread_mutex_lock( &mutexWriteRCritical );
//         sort(critValues.begin(), critValues.end(), lessCompare);
//         writeRCriticalValues(settings.outputBaseName+"-temp.dat", critValues, settings);
//         writeTSMatrix(settings.outputBaseName+"-FCRanks-temp.dat", "Feldman Cousins Ranks", fcRankMatrix, settings);
//         writeTSMatrix(settings.outputBaseName+"-SiganFitValues-temp.dat", "Best Fit Values", muFitMatrix, settings);
//         pthread_mutex_unlock(&mutexWriteRCritical);
//         //===UnLock Write r-crit===
//
//
//         //===Lock Fetch Mu===
//         pthread_mutex_lock( &mutexFetchHypothesis);
//         muQueueEmpty = mu.empty();
//         pthread_mutex_unlock( &mutexFetchHypothesis );
//         //===UnLock Fetch Mu===
//     }


    for(uint32_t i = 0; i < nThreads; i++){
        pthread_join( threadHandels[i], NULL);
    }
}
//_____________________________________________________________________________
void FeldmanCousinsAnalysis::GenerateLimitsEnsemble(double xi, bn::ndarray &up, bn::ndarray &down, uint64_t nExperiments, double cl ){
    if(cl != -1)
        cl_ = cl;
    ranks_.SetConfidenceLevel(cl_);
    if(!bn::dtype::equivalent(up.get_dtype(),bn::dtype::get_builtin<double>()))
        throw std::invalid_argument("dtype of the up numpy array must be double");
    if(!bn::dtype::equivalent(down.get_dtype(),bn::dtype::get_builtin<double>()))
        throw std::invalid_argument("dtype of the down numpy array must be double");
    if(up.shape(0)!=nExperiments)
        throw std::invalid_argument("Size of the up numpy array does not match nExperiments");
    if(down.shape(0)!=nExperiments)
        throw std::invalid_argument("Size of the down numpy array does not match nExperiments");
    double *upit = reinterpret_cast<double*>( up.get_data());
    double *downit = reinterpret_cast<double*>(down.get_data());
    double res_up, res_down;

    for(uint64_t i = 0; i < nExperiments; i++,upit++,downit++){
            Sample(xi);
            ComputeLimits(res_up, res_down);

            *upit =res_up; //.getitem(i) = res_up;
            *downit = res_down;
    }

}
//_____________________________________________________________________________
void *rankComputationThread(void *data){
    //Grabbing the data in a convienient pointer
    FCThreadData *fcThreadData = (FCThreadData *) data;
    FeldmanCousinsAnalysis &analysis = *fcThreadData->ana;


    vector<double> ranks;
    vector<double> muFits;
    double currentHypothesis = 0;
    double currentRankAtCL = 0;


    //Fetch first mu value to be computed
    //===Lock Fetch Hypothesis===++++++++++++++++++++++++
    pthread_mutex_lock( &mutexFetchHypothesis );
    cout<<"Starting Thread:"<<setw(2)<<fcThreadData->threadNumber<<endl;
    bool muQueueEmpty = fcThreadData->testHypothesisSet.empty();
    if(!muQueueEmpty){
        currentHypothesis = fcThreadData->testHypothesisSet.front();
        fcThreadData->testHypothesisSet.pop();
    }
    pthread_mutex_unlock( &mutexFetchHypothesis );
    //===UnLock Fetch Hypothesis===----------------------


    uint64_t n = fcThreadData->nExperiments;
    ranks.resize(n);
    double cl = 0.9;
    while(!muQueueEmpty){

        for(uint64_t i = 0; i < n; i++){
            analysis.Sample(currentHypothesis);
            ranks[i] = analysis.EvaluateTestsStatistic(currentHypothesis);
        }
        std::sort(ranks.begin(), ranks.end());
        currentRankAtCL = ranks[(1-cl)*(n-1)];
        //===Lock Write Ranks===++++++++++++++++
        pthread_mutex_lock( &mutexWriteRanks );
        fcThreadData->nTestedHypotheses++;
        cout<<"=====Thread:"<<setw(2)<<fcThreadData->threadNumber<<"====="<<endl;
        cout<<"Signal :"<<currentHypothesis<<endl;
        cout<<"Rank (@"<<cl*100<<"%CL and "<<n<<" experiments): "<<currentRankAtCL<<endl;
        cout<<"Hypotheses tested "<<fcThreadData->nTestedHypotheses<<"/"<<fcThreadData->totalHypotheses<<endl;
        cout<<"==================="<<endl;

        //Transfering the computed ranks to the global ranks object in the main thread
        fcThreadData->globalRanks.Fill(currentHypothesis,ranks,false);

        pthread_mutex_unlock( &mutexWriteRanks );
        //===UnLock Write Ranks===--------------

        //===Lock Fetch Hypothesis===+++++++++++++++++++
        pthread_mutex_lock( &mutexFetchHypothesis );

        muQueueEmpty = fcThreadData->testHypothesisSet.empty();
        if(!muQueueEmpty){
            currentHypothesis = fcThreadData->testHypothesisSet.front();
            fcThreadData->testHypothesisSet.pop();
        }

        pthread_mutex_unlock( &mutexFetchHypothesis );
        //===UnLock Fetch Hypothesis===-----------------

    }


    return 0;
}
