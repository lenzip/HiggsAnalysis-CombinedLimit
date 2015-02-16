#include "../interface/MultiDimFit.h"
#include <stdexcept>
#include <cmath>

#include "TMath.h"
#include "TH1.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooRandom.h"
#include "RooAbsData.h"
#include "RooAddPdf.h"
#include "RooFitResult.h"
#include "RooSimultaneous.h"
#include "RooProdPdf.h"
#include "../interface/RooMinimizerOpt.h"
#include <RooStats/ModelConfig.h>
#include "../interface/Combine.h"
#include "../interface/CascadeMinimizer.h"
#include "../interface/CloseCoutSentry.h"
#include "../interface/utils.h"
#include "../interface/MaxLikelihoodFit.h"

#include <Math/Minimizer.h>
#include <Math/MinimizerOptions.h>
#include <Math/QuantFuncMathCore.h>
#include <Math/ProbFunc.h>

using namespace RooStats;

MultiDimFit::Algo MultiDimFit::algo_ = None;
MultiDimFit::GridType MultiDimFit::gridType_ = G1x1;
std::vector<std::string>  MultiDimFit::poi_;
std::vector<RooRealVar *> MultiDimFit::poiVars_;
std::vector<float>        MultiDimFit::poiVals_;
RooArgList                MultiDimFit::poiList_;
float                     MultiDimFit::deltaNLL_ = 0;
unsigned int MultiDimFit::points_ = 50;
unsigned int MultiDimFit::firstPoint_ = 0;
unsigned int MultiDimFit::lastPoint_  = std::numeric_limits<unsigned int>::max();
bool MultiDimFit::floatOtherPOIs_ = false;
unsigned int MultiDimFit::nOtherFloatingPoi_ = 0;
bool MultiDimFit::fastScan_ = false;
bool MultiDimFit::loadedSnapshot_ = false;
bool MultiDimFit::hasMaxDeltaNLLForProf_ = false;
bool MultiDimFit::squareDistPoiStep_ = false;
float MultiDimFit::maxDeltaNLLForProf_ = 200;


MultiDimFit::MultiDimFit() :
    FitterAlgoBase("MultiDimFit specific options")
{
    options_.add_options()
        ("algo",  boost::program_options::value<std::string>()->default_value("none"), "Algorithm to compute uncertainties")
        ("poi,P",   boost::program_options::value<std::vector<std::string> >(&poi_), "Parameters of interest to fit (default = all)")
        ("floatOtherPOIs",   boost::program_options::value<bool>(&floatOtherPOIs_)->default_value(floatOtherPOIs_), "POIs other than the selected ones will be kept freely floating (1) or fixed (0, default)")
        ("squareDistPoiStep","POI step size based on distance from midpoint (max-min)/2 rather than linear")
        ("points",  boost::program_options::value<unsigned int>(&points_)->default_value(points_), "Points to use for grid or contour scans")
        ("firstPoint",  boost::program_options::value<unsigned int>(&firstPoint_)->default_value(firstPoint_), "First point to use")
        ("lastPoint",  boost::program_options::value<unsigned int>(&lastPoint_)->default_value(lastPoint_), "Last point to use")
        ("fastScan", "Do a fast scan, evaluating the likelihood without profiling it.")
        ("maxDeltaNLLForProf",  boost::program_options::value<float>(&maxDeltaNLLForProf_)->default_value(maxDeltaNLLForProf_), "Last point to use")
       ;
}

void MultiDimFit::applyOptions(const boost::program_options::variables_map &vm) 
{
    applyOptionsBase(vm);
    std::string algo = vm["algo"].as<std::string>();
    if (algo == "none") {
        algo_ = None;
    } else if (algo == "singles") {
        algo_ = Singles;
    } else if (algo == "cross") {
        algo_ = Cross;
    } else if (algo == "grid" || algo == "grid3x3" ) {
        algo_ = Grid; gridType_ = G1x1;
        if (algo == "grid3x3") gridType_ = G3x3;
    } else if (algo == "random") {
        algo_ = RandomPoints;
    } else if (algo == "contour2d") {
        algo_ = Contour2D;
    } else if (algo == "stitch2d") {
        algo_ = Stitch2D;
    } else throw std::invalid_argument(std::string("Unknown algorithm: "+algo));
    fastScan_ = (vm.count("fastScan") > 0);
    squareDistPoiStep_ = (vm.count("squareDistPoiStep") > 0);
    hasMaxDeltaNLLForProf_ = !vm["maxDeltaNLLForProf"].defaulted();
    loadedSnapshot_ = !vm["snapshotName"].defaulted();
}

bool MultiDimFit::runSpecific(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) { 
    // one-time initialization of POI variables, TTree branches, ...
    static int isInit = false;
    if (!isInit) { initOnce(w, mc_s); isInit = true; }

    fitOut.reset(TFile::Open("./mlfit.root", "RECREATE"));
    createFitResultTrees(*mc_s,withSystematics);

    // Get PDF
    RooAbsPdf &pdf = *mc_s->GetPdf();

    // Process POI not in list
    nOtherFloatingPoi_ = 0;
    RooLinkedListIter iterP = mc_s->GetParametersOfInterest()->iterator();
    for (RooAbsArg *a = (RooAbsArg*) iterP.Next(); a != 0; a = (RooAbsArg*) iterP.Next()) {
        if (poiList_.contains(*a)) continue;
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv == 0) { std::cerr << "MultiDimFit: Parameter of interest " << a->GetName() << " which is not a RooRealVar will be ignored" << std::endl; continue; }
        rrv->setConstant(!floatOtherPOIs_);
        if (floatOtherPOIs_) nOtherFloatingPoi_++;
    }
 
    // start with a best fit
    const RooCmdArg &constrainCmdArg = withSystematics  ? RooFit::Constrain(*mc_s->GetNuisanceParameters()) : RooCmdArg();
    std::auto_ptr<RooFitResult> res;
    if ( algo_ == Singles || !loadedSnapshot_ ){
    	res.reset(doFit(pdf, data, (algo_ == Singles ? poiList_ : RooArgList()), constrainCmdArg, false, 1, true, false)); 
    }
    if ( loadedSnapshot_ || res.get() || keepFailures_) {
        for (int i = 0, n = poi_.size(); i < n; ++i) {
            poiVals_[i] = poiVars_[i]->getVal();
        }
        if (algo_ != None) Combine::commitPoint(/*expected=*/false, /*quantile=*/1.); // otherwise we get it multiple times
    }
   
    std::auto_ptr<RooAbsReal> nll;
    if (algo_ != None && algo_ != Singles) {
        nll.reset(pdf.createNLL(data, constrainCmdArg, RooFit::Extended(pdf.canBeExtended())));
    } 
    
    //set snapshot for best fit
    if (!loadedSnapshot_) w->saveSnapshot("MultiDimFit",w->allVars());
    
    switch(algo_) {
        case None: 
            if (verbose > 0) {
                std::cout << "\n --- MultiDimFit ---" << std::endl;
                std::cout << "best fit parameter values: "  << std::endl;
                int len = poi_[0].length();
                for (int i = 0, n = poi_.size(); i < n; ++i) {
                    len = std::max<int>(len, poi_[i].length());
                }
                for (int i = 0, n = poi_.size(); i < n; ++i) {
                    printf("   %*s :  %+8.3f\n", len, poi_[i].c_str(), poiVals_[i]);
                }
            }
            break;
        case Singles: if (res.get()) doSingles(*res); break;
        case Cross: doBox(*nll, cl, "box", true); break;
        case Grid: doGrid(*nll); break;
        case RandomPoints: doRandomPoints(*nll); break;
        case Contour2D: doContour2D(*nll); break;
        case Stitch2D: doStitch2D(*nll); break;
    }

    fitStatus_ = res->status();
    numbadnll_ = res->numInvalidNLL();

    if (withSystematics)   {
      setFitResultTrees(mc_s->GetNuisanceParameters(),nuisanceParameters_);
      setFitResultTrees(mc_s->GetGlobalObservables(),globalObservables_);
      setFitResultTrees(mc_s->GetParametersOfInterest(),mu_);
    }
                 

    if (1) {
          RooArgSet *norms = new RooArgSet();
          norms->setName("norm_fit_s");
          MaxLikelihoodFit::CovarianceReSampler sampler(res.get());
          getNormalizations(mc_s->GetPdf(), *mc_s->GetObservables(), *norms, sampler, fitOut.get(), "_fit_s");
          setNormsFitResultTrees(norms,processNormalizations_);
          delete norms;
    }
   
    if (t_fit_sb_) t_fit_sb_->Fill();
    fitOut->WriteTObject(res.get(),"fit_s");

    if (fitOut.get()) {
      fitOut->cd();
      t_fit_sb_->Write();
      fitOut.release()->Close();
    }

    return true;
}

void MultiDimFit::initOnce(RooWorkspace *w, RooStats::ModelConfig *mc_s) {
    RooArgSet mcPoi(*mc_s->GetParametersOfInterest());
    if (poi_.empty()) {
        RooLinkedListIter iterP = mc_s->GetParametersOfInterest()->iterator();
        for (RooAbsArg *a = (RooAbsArg*) iterP.Next(); a != 0; a = (RooAbsArg*) iterP.Next()) {
            poi_.push_back(a->GetName());
        }
    }
    for (std::vector<std::string>::const_iterator it = poi_.begin(), ed = poi_.end(); it != ed; ++it) {
        RooAbsArg *a = mcPoi.find(it->c_str());
        if (a == 0) throw std::invalid_argument(std::string("Parameter of interest ")+*it+" not in model.");
        RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
        if (rrv == 0) throw std::invalid_argument(std::string("Parameter of interest ")+*it+" not a RooRealVar.");
        poiVars_.push_back(rrv);
        poiVals_.push_back(rrv->getVal());
        poiList_.add(*rrv);
    }
    // then add the branches to the tree (at the end, so there are no resizes)
    for (int i = 0, n = poi_.size(); i < n; ++i) {
        Combine::addBranch(poi_[i].c_str(), &poiVals_[i], (poi_[i]+"/F").c_str()); 
    }
    Combine::addBranch("deltaNLL", &deltaNLL_, "deltaNLL/F");
}

void MultiDimFit::doSingles(RooFitResult &res)
{
    std::cout << "\n --- MultiDimFit ---" << std::endl;
    std::cout << "best fit parameter values and profile-likelihood uncertainties: "  << std::endl;
    int len = poi_[0].length();
    for (int i = 0, n = poi_.size(); i < n; ++i) {
        len = std::max<int>(len, poi_[i].length());
    }
    for (int i = 0, n = poi_.size(); i < n; ++i) {
	RooAbsArg *rfloat = res.floatParsFinal().find(poi_[i].c_str());
	if (!rfloat) {
		rfloat = res.constPars().find(poi_[i].c_str());
	}
        RooRealVar *rf = dynamic_cast<RooRealVar*>(rfloat);
        double bestFitVal = rf->getVal();

        double hiErr = +(rf->hasRange("err68") ? rf->getMax("err68") - bestFitVal : rf->getAsymErrorHi());
        double loErr = -(rf->hasRange("err68") ? rf->getMin("err68") - bestFitVal : rf->getAsymErrorLo());
        double maxError = std::max<double>(std::max<double>(hiErr, loErr), rf->getError());

        if (fabs(hiErr) < 0.001*maxError) hiErr = -bestFitVal + rf->getMax();
        if (fabs(loErr) < 0.001*maxError) loErr = +bestFitVal - rf->getMin();

        double hiErr95 = +(do95_ && rf->hasRange("err95") ? rf->getMax("err95") - bestFitVal : 0);
        double loErr95 = -(do95_ && rf->hasRange("err95") ? rf->getMin("err95") - bestFitVal : 0);

        poiVals_[i] = bestFitVal - loErr; Combine::commitPoint(true, /*quantile=*/0.32);
        poiVals_[i] = bestFitVal + hiErr; Combine::commitPoint(true, /*quantile=*/0.32);
        if (do95_ && rf->hasRange("err95")) {
            poiVals_[i] = rf->getMax("err95"); Combine::commitPoint(true, /*quantile=*/0.05);
            poiVals_[i] = rf->getMin("err95"); Combine::commitPoint(true, /*quantile=*/0.05);
            poiVals_[i] = bestFitVal;
            printf("   %*s :  %+8.3f   %+6.3f/%+6.3f (68%%)    %+6.3f/%+6.3f (95%%) \n", len, poi_[i].c_str(), 
                    poiVals_[i], -loErr, hiErr, loErr95, -hiErr95);
        } else {
            poiVals_[i] = bestFitVal;
            printf("   %*s :  %+8.3f   %+6.3f/%+6.3f (68%%)\n", len, poi_[i].c_str(), 
                    poiVals_[i], -loErr, hiErr);
        }
    }
}

void MultiDimFit::doGrid(RooAbsReal &nll) 
{
    unsigned int n = poi_.size();
    //if (poi_.size() > 2) throw std::logic_error("Don't know how to do a grid with more than 2 POIs.");
    double nll0 = nll.getVal();

    std::vector<double> p0(n), pmin(n), pmax(n);
    for (unsigned int i = 0; i < n; ++i) {
        p0[i] = poiVars_[i]->getVal();
        pmin[i] = poiVars_[i]->getMin();
        pmax[i] = poiVars_[i]->getMax();
        poiVars_[i]->setConstant(true);
    }

    CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
    minim.setStrategy(minimizerStrategy_);
    std::auto_ptr<RooArgSet> params(nll.getParameters((const RooArgSet *)0));
    RooArgSet snap; params->snapshot(snap);
    //snap.Print("V");
    if (n == 1) {
	// can do a more intellegent spacing of points
        for (unsigned int i = 0; i < points_; ++i) {
            if (i < firstPoint_) continue;
            if (i > lastPoint_)  break;
            double x =  pmin[0] + (i+0.5)*(pmax[0]-pmin[0])/points_; 
	    if (squareDistPoiStep_){
		// distance between steps goes as ~square of distance from middle or range (could this be changed to from best fit value?)
		double phalf = (pmax[0]-pmin[0])/2;
		if (i<(unsigned int)points_/2) x = pmin[0]+TMath::Sqrt(2*i*(phalf)*(phalf)/points_);
		else x = pmax[0]-TMath::Sqrt(2*(points_-i)*(phalf)*(phalf)/points_);
	    }

            if (verbose > 1) std::cout << "Point " << i << "/" << points_ << " " << poiVars_[0]->GetName() << " = " << x << std::endl;
            *params = snap; 
            poiVals_[0] = x;
            poiVars_[0]->setVal(x);
            // now we minimize
            bool ok = fastScan_ || (hasMaxDeltaNLLForProf_ && (nll.getVal() - nll0) > maxDeltaNLLForProf_) ? 
                        true : 
                        minim.minimize(verbose-1);
            if (ok) {
                deltaNLL_ = nll.getVal() - nll0;
                double qN = 2*(deltaNLL_);
                double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                Combine::commitPoint(true, /*quantile=*/prob);
            }
        }
    } else if (n == 2) {
        unsigned int sqrn = ceil(sqrt(double(points_)));
        unsigned int ipoint = 0, nprint = ceil(0.005*sqrn*sqrn);
        RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CountErrors);
        CloseCoutSentry sentry(verbose < 2);
        double deltaX =  (pmax[0]-pmin[0])/sqrn, deltaY = (pmax[1]-pmin[1])/sqrn;
        for (unsigned int i = 0; i < sqrn; ++i) {
            for (unsigned int j = 0; j < sqrn; ++j, ++ipoint) {
                if (ipoint < firstPoint_) continue;
                if (ipoint > lastPoint_)  break;
                *params = snap; 
                double x =  pmin[0] + (i+0.5)*deltaX; 
                double y =  pmin[1] + (j+0.5)*deltaY; 
                if (verbose && (ipoint % nprint == 0)) {
                         fprintf(sentry.trueStdOut(), "Point %d/%d, (i,j) = (%d,%d), %s = %f, %s = %f\n",
                                        ipoint,sqrn*sqrn, i,j, poiVars_[0]->GetName(), x, poiVars_[1]->GetName(), y);
                }
                poiVals_[0] = x;
                poiVals_[1] = y;
                poiVars_[0]->setVal(x);
                poiVars_[1]->setVal(y);
                nll.clearEvalErrorLog(); nll.getVal();
                if (nll.numEvalErrors() > 0) { 
                    deltaNLL_ = 9999; Combine::commitPoint(true, /*quantile=*/0); 
                    if (gridType_ == G3x3) {
                        for (int i2 = -1; i2 <= +1; ++i2) {
                            for (int j2 = -1; j2 <= +1; ++j2) {
                                if (i2 == 0 && j2 == 0) continue;
                                poiVals_[0] = x + 0.33333333*i2*deltaX;
                                poiVals_[1] = y + 0.33333333*j2*deltaY;
                                deltaNLL_ = 9999; Combine::commitPoint(true, /*quantile=*/0); 
                            }
                        }
                    }
                    continue;
                }
                // now we minimize
                bool skipme = hasMaxDeltaNLLForProf_ && (nll.getVal() - nll0) > maxDeltaNLLForProf_;
                bool ok = fastScan_ || skipme ? true :  minim.minimize(verbose-1);
                if (ok) {
                    deltaNLL_ = nll.getVal() - nll0;
                    double qN = 2*(deltaNLL_);
                    double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                    Combine::commitPoint(true, /*quantile=*/prob);
                }
                if (gridType_ == G3x3) {
                    bool forceProfile = !fastScan_ && std::min(fabs(deltaNLL_ - 1.15), fabs(deltaNLL_ - 2.995)) < 0.5;
                    utils::CheapValueSnapshot center(*params);
                    double x0 = x, y0 = y;
                    for (int i2 = -1; i2 <= +1; ++i2) {
                        for (int j2 = -1; j2 <= +1; ++j2) {
                            if (i2 == 0 && j2 == 0) continue;
                            center.writeTo(*params);
                            x = x0 + 0.33333333*i2*deltaX;
                            y = y0 + 0.33333333*j2*deltaY;
                            poiVals_[0] = x; poiVars_[0]->setVal(x);
                            poiVals_[1] = y; poiVars_[1]->setVal(y);
                            nll.clearEvalErrorLog(); nll.getVal();
                            if (nll.numEvalErrors() > 0) { 
                                deltaNLL_ = 9999; Combine::commitPoint(true, /*quantile=*/0); 
                                continue;
                            }
                            deltaNLL_ = nll.getVal() - nll0;
                            if (forceProfile || (!fastScan_ && std::min(fabs(deltaNLL_ - 1.15), fabs(deltaNLL_ - 2.995)) < 0.5)) {
                                minim.minimize(verbose-1);
                                deltaNLL_ = nll.getVal() - nll0;
                            }
                            double qN = 2*(deltaNLL_);
                            double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                            Combine::commitPoint(true, /*quantile=*/prob);
                        }
                    }
                }
            }
        }

    } else { // Use utils routine if n > 2 

        unsigned int rootn = ceil(TMath::Power(double(points_),double(1./n)));
        unsigned int ipoint = 0, nprint = ceil(0.005*TMath::Power((double)rootn,(double)n));
	
        RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CountErrors);
        CloseCoutSentry sentry(verbose < 2);
	
	// Create permutations 
        std::vector<int> axis_points;
	
        for (unsigned int poi_i=0;poi_i<n;poi_i++){
	  axis_points.push_back((int)rootn);
    	}

        std::vector<std::vector<int> > permutations = utils::generateCombinations(axis_points);
	// Step through points
        std::vector<std::vector<int> >::iterator perm_it = permutations.begin();
	int npermutations = permutations.size();
    	for (;perm_it!=permutations.end(); perm_it++){

          if (ipoint < firstPoint_) continue;
          if (ipoint > lastPoint_)  break;
          *params = snap; 

          if (verbose && (ipoint % nprint == 0)) {
             fprintf(sentry.trueStdOut(), "Point %d/%d, ",
                          ipoint,npermutations);
          }	  
          for (unsigned int poi_i=0;poi_i<n;poi_i++){
	    int ip = (*perm_it)[poi_i];
            double deltaXi = (pmax[poi_i]-pmin[poi_i])/rootn;
	    double xi = pmin[poi_i]+deltaXi*(ip+0.5);
            poiVals_[poi_i] = xi; poiVars_[poi_i]->setVal(xi);
	    if (verbose && (ipoint % nprint == 0)){
             fprintf(sentry.trueStdOut(), " %s = %f ",
                          poiVars_[poi_i]->GetName(), xi);
	    }
	  }
	  if (verbose && (ipoint % nprint == 0)) fprintf(sentry.trueStdOut(), "\n");

          nll.clearEvalErrorLog(); nll.getVal();
          if (nll.numEvalErrors() > 0) { 
               deltaNLL_ = 9999; Combine::commitPoint(true, /*quantile=*/0);
	       continue;
	  }
          // now we minimize
          bool skipme = hasMaxDeltaNLLForProf_ && (nll.getVal() - nll0) > maxDeltaNLLForProf_;
          bool ok = fastScan_ || skipme ? true :  minim.minimize(verbose-1);
          if (ok) {
               deltaNLL_ = nll.getVal() - nll0;
               double qN = 2*(deltaNLL_);
               double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
               Combine::commitPoint(true, /*quantile=*/prob);
          }
	  ipoint++;	
	} 
    }
}

void MultiDimFit::doRandomPoints(RooAbsReal &nll) 
{
    double nll0 = nll.getVal();
    for (unsigned int i = 0, n = poi_.size(); i < n; ++i) {
        poiVars_[i]->setConstant(true);
    }

    CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
    minim.setStrategy(minimizerStrategy_);
    unsigned int n = poi_.size();
    for (unsigned int j = 0; j < points_; ++j) {
        for (unsigned int i = 0; i < n; ++i) {
            poiVars_[i]->randomize();
            poiVals_[i] = poiVars_[i]->getVal(); 
        }
        // now we minimize
        {   
            CloseCoutSentry sentry(verbose < 3);    
            bool ok = minim.minimize(verbose-1);
            if (ok) {
                double qN = 2*(nll.getVal() - nll0);
                double prob = ROOT::Math::chisquared_cdf_c(qN, n+nOtherFloatingPoi_);
                Combine::commitPoint(true, /*quantile=*/prob);
            }
        } 
    }
}

void MultiDimFit::doContour2D(RooAbsReal &nll) 
{
    if (poi_.size() != 2) throw std::logic_error("Contour2D works only in 2 dimensions");
    RooRealVar *xv = poiVars_[0]; double x0 = poiVals_[0]; float &x = poiVals_[0];
    RooRealVar *yv = poiVars_[1]; double y0 = poiVals_[1]; float &y = poiVals_[1];

    double threshold = nll.getVal() + 0.5*ROOT::Math::chisquared_quantile_c(1-cl,2+nOtherFloatingPoi_);
    if (verbose>0) std::cout << "Best fit point is for " << xv->GetName() << ", "  << yv->GetName() << " =  " << x0 << ", " << y0 << std::endl;

    // make a box
    doBox(nll, cl, "box");
    double xMin = xv->getMin("box"), xMax = xv->getMax("box");
    double yMin = yv->getMin("box"), yMax = yv->getMax("box");

    verbose--; // reduce verbosity to avoid messages from findCrossing
    // ===== Get relative min/max of x for several fixed y values =====
    yv->setConstant(true);
    for (unsigned int j = 0; j <= points_; ++j) {
        if (j < firstPoint_) continue;
        if (j > lastPoint_)  break;
        // take points uniformly spaced in polar angle in the case of a perfect circle
        double yc = 0.5*(yMax + yMin), yr = 0.5*(yMax - yMin);
        yv->setVal( yc + yr * std::cos(j*M_PI/double(points_)) );
        // ===== Get the best fit x (could also do without profiling??) =====
        xv->setConstant(false);  xv->setVal(x0);
        CascadeMinimizer minimXI(nll, CascadeMinimizer::Unconstrained, xv);
        minimXI.setStrategy(minimizerStrategy_);
        {
            CloseCoutSentry sentry(verbose < 3);    
            minimXI.minimize(verbose-1);
        }
        double xc = xv->getVal(); xv->setConstant(true);
        if (verbose>-1) std::cout << "Best fit " << xv->GetName() << " for  " << yv->GetName() << " = " << yv->getVal() << " is at " << xc << std::endl;
        // ===== Then get the range =====
        CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
        double xup = findCrossing(minim, nll, *xv, threshold, xc, xMax);
        if (!std::isnan(xup)) { 
            x = xup; y = yv->getVal(); Combine::commitPoint(true, /*quantile=*/1-cl);
            if (verbose>-1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for " << yv->GetName() << " = " << y << " is " << x << std::endl;
        }
        
        double xdn = findCrossing(minim, nll, *xv, threshold, xc, xMin);
        if (!std::isnan(xdn)) { 
            x = xdn; y = yv->getVal(); Combine::commitPoint(true, /*quantile=*/1-cl);
            if (verbose>-1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for " << yv->GetName() << " = " << y << " is " << x << std::endl;
        }
    }

    verbose++; // restore verbosity
}

void MultiDimFit::doStitch2D(RooAbsReal &nll)
{
    if (poi_.size() != 2) throw std::logic_error("Contour2D works only in 2 dimensions");
    //RooRealVar *xv = poiVars_[0]; double x0 = poiVals_[0]; float &x = poiVals_[0];
    //RooRealVar *yv = poiVars_[1]; double y0 = poiVals_[1]; float &y = poiVals_[1];

    //double threshold = nll.getVal() + 0.5*ROOT::Math::chisquared_quantile_c(1-cl,2+nOtherFloatingPoi_);
    //if (verbose>0) std::cout << "Best fit point is for " << xv->GetName() << ", "  << yv->GetName() << " =  " << x0 << ", " << y0 << std::endl;

    // make a box
    //doBox(nll, cl, "box");
    //double xMin = xv->getMin("box"), xMax = xv->getMax("box");
    //double yMin = yv->getMin("box"), yMax = yv->getMax("box");

//    verbose--; // reduce verbosity to avoid messages from findCrossing
//    // ===== Get relative min/max of x for several fixed y values =====
//    yv->setConstant(true);
//    for (unsigned int j = 0; j <= points_; ++j) {
//        if (j < firstPoint_) continue;
//        if (j > lastPoint_)  break;
//        // take points uniformly spaced in polar angle in the case of a perfect circle
//        double yc = 0.5*(yMax + yMin), yr = 0.5*(yMax - yMin);
//        yv->setVal( yc + yr * std::cos(j*M_PI/double(points_)) );
//        // ===== Get the best fit x (could also do without profiling??) =====
//        xv->setConstant(false);  xv->setVal(x0);
//        CascadeMinimizer minimXI(nll, CascadeMinimizer::Unconstrained, xv);
//        minimXI.setStrategy(minimizerStrategy_);
//        {
//            CloseCoutSentry sentry(verbose < 3);
//            minimXI.minimize(verbose-1);
//        }
//        double xc = xv->getVal(); xv->setConstant(true);
//        if (verbose>-1) std::cout << "Best fit " << xv->GetName() << " for  " << yv->GetName() << " = " << yv->getVal() << " is at " << xc << std::endl;
//        // ===== Then get the range =====
//        CascadeMinimizer minim(nll, CascadeMinimizer::Constrained);
//        double xup = findCrossing(minim, nll, *xv, threshold, xc, xMax);
//        if (!std::isnan(xup)) {
//            x = xup; y = yv->getVal(); Combine::commitPoint(true, /*quantile=*/1-cl);
//            if (verbose>-1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for " << yv->GetName() << " = " << y << " is " << x << std::endl;
//        }
//
//        double xdn = findCrossing(minim, nll, *xv, threshold, xc, xMin);
//        if (!std::isnan(xdn)) {
//            x = xdn; y = yv->getVal(); Combine::commitPoint(true, /*quantile=*/1-cl);
//            if (verbose>-1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for " << yv->GetName() << " = " << y << " is " << x << std::endl;
//        }
//    }
//
//    verbose++; // restore verbosity
}

void MultiDimFit::getShapesAndNorms(RooAbsPdf *pdf, const RooArgSet &obs, std::map<std::string,MaxLikelihoodFit::ShapeAndNorm> &out, const std::string &channel) {
    RooSimultaneous *sim = dynamic_cast<RooSimultaneous *>(pdf);
    if (sim != 0) {
        RooAbsCategoryLValue &cat = const_cast<RooAbsCategoryLValue &>(sim->indexCat());
        for (int i = 0, n = cat.numBins((const char *)0); i < n; ++i) {
            cat.setBin(i);
            RooAbsPdf *pdfi = sim->getPdf(cat.getLabel());
            if (pdfi) getShapesAndNorms(pdfi, obs, out, cat.getLabel());
        }
        return;
    }
    RooProdPdf *prod = dynamic_cast<RooProdPdf *>(pdf);
    if (prod != 0) {
        RooArgList list(prod->pdfList());
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            if (pdfi->dependsOn(obs)) getShapesAndNorms(pdfi, obs, out, channel);
        }
        return;
    }
    RooAddPdf *add = dynamic_cast<RooAddPdf *>(pdf);
    if (add != 0) {
        RooArgList clist(add->coefList());
        RooArgList plist(add->pdfList());
        for (int i = 0, n = clist.getSize(); i < n; ++i) {
            RooAbsReal *coeff = (RooAbsReal *) clist.at(i);
            MaxLikelihoodFit::ShapeAndNorm &ns = out[coeff->GetName()];
            ns.norm = coeff;
            ns.pdf = (RooAbsPdf*) plist.at(i);
            ns.channel = (coeff->getStringAttribute("combine.channel") ? coeff->getStringAttribute("combine.channel") : channel.c_str());
            ns.process = (coeff->getStringAttribute("combine.process") ? coeff->getStringAttribute("combine.process") : ns.norm->GetName());
            ns.signal  = (coeff->getStringAttribute("combine.process") ? coeff->getAttribute("combine.signal") : (strstr(ns.norm->GetName(),"shapeSig") != 0));
            std::auto_ptr<RooArgSet> myobs(ns.pdf->getObservables(obs));
            ns.obs.add(*myobs);
        }
        return;
    }
}


void MultiDimFit::doBox(RooAbsReal &nll, double cl, const char *name, bool commitPoints)  {
    unsigned int n = poi_.size();
    double nll0 = nll.getVal(), threshold = nll0 + 0.5*ROOT::Math::chisquared_quantile_c(1-cl,n+nOtherFloatingPoi_);

    std::vector<double> p0(n);
    for (unsigned int i = 0; i < n; ++i) {
        p0[i] = poiVars_[i]->getVal();
        poiVars_[i]->setConstant(false);
    }

    verbose--; // reduce verbosity due to findCrossing
    for (unsigned int i = 0; i < n; ++i) {
        RooRealVar *xv = poiVars_[i];
        xv->setConstant(true);
        CascadeMinimizer minimX(nll, CascadeMinimizer::Constrained);
        minimX.setStrategy(minimizerStrategy_);

        for (unsigned int j = 0; j < n; ++j) poiVars_[j]->setVal(p0[j]);
        double xMin = findCrossing(minimX, nll, *xv, threshold, p0[i], xv->getMin()); 
        if (!std::isnan(xMin)) { 
            if (verbose > -1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMin << std::endl;
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/1-cl);
        } else {
            xMin = xv->getMin();
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            double prob = ROOT::Math::chisquared_cdf_c(2*(nll.getVal() - nll0), n+nOtherFloatingPoi_);
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/prob);
            if (verbose > -1) std::cout << "Minimum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMin << " (on the boundary, p-val " << prob << ")" << std::endl;
        }
        
        for (unsigned int j = 0; j < n; ++j) poiVars_[j]->setVal(p0[j]);
        double xMax = findCrossing(minimX, nll, *xv, threshold, p0[i], xv->getMax()); 
        if (!std::isnan(xMax)) { 
            if (verbose > -1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMax << std::endl;
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/1-cl);
        } else {
            xMax = xv->getMax();
            double prob = ROOT::Math::chisquared_cdf_c(2*(nll.getVal() - nll0), n+nOtherFloatingPoi_);
            for (unsigned int j = 0; j < n; ++j) poiVals_[j] = poiVars_[j]->getVal();
            if (commitPoints) Combine::commitPoint(true, /*quantile=*/prob);
            if (verbose > -1) std::cout << "Maximum of " << xv->GetName() << " at " << cl << " CL for all others floating is " << xMax << " (on the boundary, p-val " << prob << ")" << std::endl;
        }

        xv->setRange(name, xMin, xMax);
        xv->setConstant(false);
    }
    verbose++; // restore verbosity 
}

void MultiDimFit::getNormalizations(RooAbsPdf *pdf, const RooArgSet &obs, RooArgSet &out, MaxLikelihoodFit::NuisanceSampler & sampler, TDirectory *fOut, const std::string &postfix) {
    // fill in a map
    std::map<std::string,MaxLikelihoodFit::ShapeAndNorm> snm;
    getShapesAndNorms(pdf,obs, snm, "");
    typedef std::map<std::string,MaxLikelihoodFit::ShapeAndNorm>::const_iterator IT;
    typedef std::map<std::string,TH1*>::const_iterator IH;
    // create directory structure for shapes
    TDirectory *shapeDir = fOut ? fOut->mkdir((std::string("shapes")+postfix).c_str()) : 0;
    std::map<std::string,TDirectory*> shapesByChannel;
    if (shapeDir) {
        for (IT it = snm.begin(), ed = snm.end(); it != ed; ++it) {
            TDirectory *& sub = shapesByChannel[it->second.channel];
            if (sub == 0) sub = shapeDir->mkdir(it->second.channel.c_str());
        }
    }
    // now let's start with the central values
    std::vector<double> vals(snm.size(), 0.), sumx2(snm.size(), 0.);
    std::vector<TH1*>   shapes(snm.size(), 0), shapes2(snm.size(), 0);
    std::vector<int>    bins(snm.size(), 0), sig(snm.size(), 0);
    std::map<std::string,TH1*> totByCh, totByCh2, sigByCh, sigByCh2, bkgByCh, bkgByCh2;
    IT bg = snm.begin(), ed = snm.end(), pair; int i;
    for (pair = bg, i = 0; pair != ed; ++pair, ++i) {
        vals[i] = pair->second.norm->getVal();
        //out.addOwned(*(new RooConstVar(pair->first.c_str(), "", pair->second.norm->getVal())));
        if (fOut != 0 && pair->second.obs.getSize() == 1) {
            RooRealVar *x = (RooRealVar*)pair->second.obs.at(0);
            TH1* hist = pair->second.pdf->createHistogram("", *x);
            hist->SetNameTitle(pair->second.process.c_str(), (pair->second.process+" in "+pair->second.channel).c_str());
            hist->Scale(vals[i] / hist->Integral("width"));
            hist->SetDirectory(shapesByChannel[pair->second.channel]);
            shapes[i] = hist;
            if (0) {
                shapes2[i] = (TH1*) hist->Clone();
                shapes2[i]->SetDirectory(0);
                shapes2[i]->Reset();
                bins[i] = hist->GetNbinsX();
                TH1 *&htot = totByCh[pair->second.channel];
                if (htot == 0) {
                    htot = (TH1*) hist->Clone();
                    htot->SetName("total");
                    htot->SetTitle(Form("Total signal+background in %s", pair->second.channel.c_str()));
                    htot->SetDirectory(shapesByChannel[pair->second.channel]);
                    TH1 *htot2 = (TH1*) hist->Clone(); htot2->Reset();
                    htot2->SetDirectory(0);
                    totByCh2[pair->second.channel] = htot2;
                } else {
                    htot->Add(hist);
                }
                sig[i] = pair->second.signal;
                TH1 *&hpart = (sig[i] ? sigByCh : bkgByCh)[pair->second.channel];
                if (hpart == 0) {
                    hpart = (TH1*) hist->Clone();
                    hpart->SetName((sig[i] ? "total_signal" : "total_background"));
                    hpart->SetTitle(Form((sig[i] ? "Total signal in %s" : "Total background in %s"),pair->second.channel.c_str()));
                    hpart->SetDirectory(shapesByChannel[pair->second.channel]);
                    TH1 *hpart2 = (TH1*) hist->Clone(); hpart2->Reset();
                    hpart2->SetDirectory(0);
                    (sig[i] ? sigByCh2 : bkgByCh2)[pair->second.channel] = hpart2;
                } else {
                    hpart->Add(hist);
                }
            }
        }
    }
    for (pair = bg, i = 0; pair != ed; ++pair, ++i) {
        RooRealVar *val = new RooRealVar((pair->second.channel+"/"+pair->second.process).c_str(), "", vals[i]);
        val->setError(sumx2[i]);
        out.addOwned(*val); 
        if (shapes[i]) shapesByChannel[pair->second.channel]->WriteTObject(shapes[i]);
    }
    if (fOut) {
        fOut->WriteTObject(&out, (std::string("norm")+postfix).c_str()); 
        for (IH h = totByCh.begin(), eh = totByCh.end(); h != eh; ++h) { shapesByChannel[h->first]->WriteTObject(h->second); }
        for (IH h = sigByCh.begin(), eh = sigByCh.end(); h != eh; ++h) { shapesByChannel[h->first]->WriteTObject(h->second); }
        for (IH h = bkgByCh.begin(), eh = bkgByCh.end(); h != eh; ++h) { shapesByChannel[h->first]->WriteTObject(h->second); }
    }
}

void MultiDimFit::setNormsFitResultTrees(const RooArgSet *args, double * vals){

         TIterator* iter(args->createIterator());
         int count=0;

         for (TObject *a = iter->Next(); a != 0; a = iter->Next()) {
                 RooRealVar *rcv = dynamic_cast<RooRealVar *>(a);
                 //std::string name = rcv->GetName();
                 vals[count]=rcv->getVal();
                 count++;
         }
         delete iter;
         return;
}

void MultiDimFit::createFitResultTrees(const RooStats::ModelConfig &mc, bool withSys){
    // Initiate the arrays to store parameters

         // create TTrees to store fit results:
         t_fit_sb_ = new TTree("tree_fit_sb","tree_fit_sb");

         t_fit_sb_->Branch("fit_status",&fitStatus_,"fit_status/Int_t");

         t_fit_sb_->Branch("numbadnll",&numbadnll_,"numbadnll/Int_t");

         //t_fit_sb_->Branch("nll_min",&nll_sb_,"nll_min/Double_t");

         //t_fit_sb_->Branch("nll_nll0",&nll_nll0_,"nll_nll0/Double_t");

         int count=0; 
         // fill the maps for the nuisances, and global observables
         RooArgSet *norms = new RooArgSet();
         getNormalizationsSimple(mc.GetPdf(), *mc.GetObservables(), *norms);
 
         processNormalizations_ = new double[norms->getSize()];
         const RooArgSet * pois = mc.GetParametersOfInterest();
         mu_ = new double[pois->getSize()];
         TIterator* iter_pois(pois->createIterator());
         for (TObject *a = iter_pois->Next(); a != 0; a = iter_pois->Next()) { 
                 RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
                 std::string name = rrv->GetName();
                 mu_[count]=0;
                 t_fit_sb_->Branch(name.c_str(),&(mu_[count]),Form("%s/Double_t",name.c_str()));
                 count++;
          }     

         // If no systematic (-S 0), then don't make nuisance trees
         if (withSys){
          count = 0;
          const RooArgSet *cons = mc.GetGlobalObservables();
          const RooArgSet *nuis = mc.GetNuisanceParameters();
          globalObservables_ = new double[cons->getSize()];
          nuisanceParameters_= new double[nuis->getSize()];

          TIterator* iter_c(cons->createIterator());
          for (TObject *a = iter_c->Next(); a != 0; a = iter_c->Next()) { 
                 RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);        
                 std::string name = rrv->GetName();
                 globalObservables_[count]=0;
                 t_fit_sb_->Branch(name.c_str(),&(globalObservables_[count]),Form("%s/Double_t",name.c_str()));
                 count++;
          }         
          count = 0;
          TIterator* iter_n(nuis->createIterator());
          for (TObject *a = iter_n->Next(); a != 0; a = iter_n->Next()) { 
                 RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);        
                 std::string name = rrv->GetName();
                 nuisanceParameters_[count] = 0;
                 t_fit_sb_->Branch(name.c_str(),&(nuisanceParameters_[count])),Form("%s/Double_t",name.c_str());
                 count++;
          }

         }

         count = 0;
         TIterator* iter_no(norms->createIterator());
         for (TObject *a = iter_no->Next(); a != 0; a = iter_no->Next()) { 
                 RooRealVar *rcv = dynamic_cast<RooRealVar *>(a);        
                 std::string name = rcv->GetName();
                 processNormalizations_[count] = 0;
                 t_fit_sb_->Branch(name.c_str(),&(processNormalizations_[count])),Form("%s/Double_t",name.c_str());
                 count++;
         }
         delete norms;

        std::cout << "Created Branches" <<std::endl;
         return;
}

void MultiDimFit::setFitResultTrees(const RooArgSet *args, double * vals){

         TIterator* iter(args->createIterator());
         int count=0;

         for (TObject *a = iter->Next(); a != 0; a = iter->Next()) {
                 RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);
                 //std::string name = rrv->GetName();
                 vals[count]=rrv->getVal();
                 count++;
         }
         delete iter;
         return;
}

void MultiDimFit::getNormalizationsSimple(RooAbsPdf *pdf, const RooArgSet &obs, RooArgSet &out) {
    RooSimultaneous *sim = dynamic_cast<RooSimultaneous *>(pdf);
    if (sim != 0) {
        RooAbsCategoryLValue &cat = const_cast<RooAbsCategoryLValue &>(sim->indexCat());
        for (int i = 0, n = cat.numBins((const char *)0); i < n; ++i) {
            cat.setBin(i);
            RooAbsPdf *pdfi = sim->getPdf(cat.getLabel());
            if (pdfi) getNormalizationsSimple(pdfi, obs, out);
        }
        return;
    }
    RooProdPdf *prod = dynamic_cast<RooProdPdf *>(pdf);
    if (prod != 0) {
        RooArgList list(prod->pdfList());
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            if (pdfi->dependsOn(obs)) getNormalizationsSimple(pdfi, obs, out);
        }
        return;
    }
    RooAddPdf *add = dynamic_cast<RooAddPdf *>(pdf);
    if (add != 0) {
        RooArgList list(add->coefList());
        for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsReal *coeff = (RooAbsReal *) list.at(i);
            out.addOwned(*(new RooRealVar(coeff->GetName(), "", coeff->getVal())));
        }
        return;
    }
}

