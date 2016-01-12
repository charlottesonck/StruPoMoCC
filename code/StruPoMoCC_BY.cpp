/*code accompanying the PhD "Structured population models with internal cell cycle" by Charlotte Sonck
  Budding yeast model of Tyson and Novák used as internal structure for the cells
  version 12 January 2016*/

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <list>
using namespace std;
#include <random>

/* include files necesarry for age integration (CVODE library)*/
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>

/*include files for matrix calculations (ALGLIB library)*/
#include <solvers.h>

#define cond_min 1 /*0 if condition of minimal mass is not used and 1 if it is used*/
#define cond_max 0 /*0 if m_max in dm/dt and 1 if m_max not in dm/dt, but in code that cell can't divide when m>m_max*/

#define NDIM 9  /*dimension chemical submodel for the cell cycle*/
#define phi 0.4  /*parameter for mass division*/
#define m_min 0.75 /*minimum mass for division*/
#define zeta1 0.5 /*parameter for dependency mass increase on nutrient*/
#define c1 2 /*parameter for dependency mass increase on nutrient*/
#define c2 1 /*parameter consumption rate*/
realtype S0=RCONST(1); /*concentration nutrient in feeding bottle*/
realtype D=RCONST(0.01); /*diffusion rate */
#define Ft 1.0e-7 /*lower threshold for survival probability F*/

/*cell cycle model parameters*/
#define k1 0.04
#define k2p 0.04
#define k2pp 1
#define k2ppp 1
#define k3p 1
#define k3pp 10
#define k4 35
#define k4p 2
#define J3 0.04
#define J4 0.04
#define k5p 0.005
#define k5pp 0.2
#define k6 0.1
#define J5 0.3
#define n 4
#define mu 0.01
#define m_max 10
#define k7 1
#define k8 0.5
#define k9 0.1
#define k10 0.02
#define k11 1
#define k12p 0.2
#define k12pp 50
#define k12ppp 100
#define k13p 0
#define k13pp 1
#define k14 1
#define k15p 1.5
#define k15pp 0.05
#define k16p 1
#define k16pp 3
#define J7 0.001
#define J8 0.001
#define J15 0.01
#define J16 0.01
#define Mad 1
#define Keq 1000
realtype X_div = RCONST(0.1); /*division threshold for X*/

/*definitions related to the age integration*/
#define T0 RCONST(0.0) /*initial age*/
#define T1 RCONST(0.5) /*first output age*/
#define RELTOL RCONST(1.0e-6) /*scalar relative tolerance*/
#define ABSTOL RCONST(1.0e-8) /*scalar absolute tolerance*/
#define NEQ 11 /*number of equations (m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,F,theta)*/
realtype TIN = RCONST(0.5); /*size agestep*/
#define eps_m RCONST(1.0e-7) /*relative to m_max, how close in the integration we maximally go to m_max*/
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data); /*functions for age integration*/
static int g(realtype t, N_Vector y, realtype *gout, void *user_data); /*functions for RootFindingFunction during age integration*/

/*definitions related to the class cohort*/
class cohort;
typedef list <cohort> clist;
typedef list <cohort>:: iterator clistit;
#define delta RCONST(1.0e-3) /*"inhibition zone" cohort*/
#define eps_b RCONST(1.0e-10) /*eps_b = epsilon = value (in %, so 1 means 1%) of total number of cells born that can be neglected*/
#define var_N RCONST(0.01) /*in function loopMap_C: relative amount of variation allowed in the number of cohorts at convergence*/
                           /* for example 0.01 means that N has to change less than 0.01*N from one iteration to the next, */
							/* in order to say that the map has converged to a fixed point*/

void integrate_allcohorts(clist coh, int out, int out2); /*integrates a given list of cohorts coh*/
void integrate_allcohorts_dest(clist coh, int out, int out2); /*integrates a given list of cohorts coh and prints data_figd3_fp.txt
															  with the information needed for the cohort-to-cohort representation*/
void addcohort(realtype p, clist &cohnew, realtype in, realtype m, realtype CycB_T, realtype Cdh1, realtype Cdc20_T, realtype Cdc20_A, realtype IEP, realtype CKI_T, realtype SK, realtype TF, int out); 
  /*adds the daughter cells of a dividing cohort (with m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF and in cells) to a list of cohorts cohnew*/
void printListCoh_OC(clist &coh); /*prints the given list of cohorts in outputcohorts.txt*/

/*global variables*/
clist cohnew; /*list of new cohorts created by integration of a list of cohorts (integrate_allcohorts)*/
ofstream myfile_OC; /*for outputfile outputcohorts.txt*/
realtype S; /*nutrient concentration*/
realtype prop = RCONST(1); /*for adjusted change in S after one age integration of a list of cohorts*/
                           /*proportion of change in S that is executed (if =1: normal adjustment)*/
realtype totaalb; /*total amount of cells born during one age integration of a list of cohorts (calculated through the cells that divide)*/
realtype totalb; /*total amount of cells born during one age integration of a list of cohorts (calculated through cohnew)*/
realtype tempS; /*to temp. save new S after integrate_allcohorts*/

/*integers used in add_cohort_dest_sub, add_cohort_dest, integrate_cohort_dest and integrate_allcohorts_dest
   to help determine to which cohorts in the list cohnew the cohorts contribute after division*/
int i_sm = 0; /*for the small daughter cells*/
int i_gr = 0; /*for the large daughter cells*/
int how_sm = 0; /*for the small daughter cells: = 1 if new cohort is created at location i_sm + 1, 
			   = 2 if cells are added to existing cohort at location i_sm, 
			   = 3 if cells are added to the cohort that is merged from the cohorts at location i_sm and i_sm+1 */
int how_gr = 0; /*for the large daughter cells*/

class cohort {
	realtype init[NDIM]; /*birth state vector*/
	realtype F; /*survival chance of cells in cohort*/
	realtype theta; /*amount of nutrient consumed by one cell in the cohort up to its current age*/
	realtype b; /*original number of cells in cohort*/
public:
	void set_init (realtype mass, realtype CycB_T, realtype Cdh1, realtype Cdc20_T, realtype Cdc20_A, realtype IEP, realtype CKI_T, realtype SK, realtype TF, realtype B); 
	          /*initializes the state vector of a cohort: mass, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF and number of cells*/
	void setb(realtype B) {b=B;}; /*initializes b (the original number of cells in the cohort)*/
	void setm(realtype m) {init[0]=m;}; /*changes the mass of the cohort*/
	void setTheta (realtype t) {theta=t;} /*changes the value of theta*/
	realtype getb() {return b;} /*returns the original number of cells*/
	realtype getTheta() {return theta;} /*returns theta*/
	realtype getMass() {return init[0];} /*returns the mass*/
	realtype getCycB_T() {return init[1];} /*returns CycB_T*/
	realtype getCdh1() {return init[2];} /*returns Cdh1*/
	realtype getCdc20_T() {return init[3];} /*returns Cdc20_T*/
	realtype getCdc20_A() {return init[4];} /*returns Cdc20_A*/
	realtype getIEP() {return init[5];} /*returns IEP*/
	realtype getCKI_T() {return init[6];} /*returns CKI_T*/
	realtype getSK() {return init[7];} /*returns SK*/
	realtype getTF() {return init[8];} /*returns TF*/
	realtype getF() {return F;} /*returns the survival probability*/
	int integrate_cohort(int out, int out2);  /*integrates the cohort until the age where F < Ft*/
	int integrate_cohort_dest(int out, int out2, int &i_sm, int &how_sm, int &i_gr, int &how_gr); /*integrates the cohort until the age where F < Ft 
																								   + gives information needed for the cohort-to-cohort-representation*/
	void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6, realtype y7, realtype y8, realtype y9, realtype y10, realtype y11);
	           /*prints the data values (t,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,F,theta) of a cohort at a certain age t in the file outputcohorts.txt*/
	void PrintOutputData(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6, realtype y7, realtype y8, realtype y9, realtype y10, realtype y11);
	           /*prints the data values (t,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,F,theta) of a cohort at a certain age t in the file values.txt*/
	int check_flag(void *flagvalue, char *funcname, int opt); /*handles CVODE errors*/
};

void cohort::set_init(realtype mass, realtype CycB_T, realtype Cdh1, realtype Cdc20_T, realtype Cdc20_A, realtype IEP, realtype CKI_T, realtype SK, realtype TF, realtype B) {
	/*initializes the state vector of a cohort: mass, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF and number of cells*/
	init[0] = mass;
	init[1] = CycB_T;
	init[2] = Cdh1;
	init[3] = Cdc20_T;
	init[4] = Cdc20_A;
	init[5] = IEP;
	init[6] = CKI_T;
	init[7] = SK;
	init[8] = TF;
	F=RCONST(1.0); //set the survival propability equal to 1 
	theta=RCONST(0.0); //set the amount of nutrient consumed by one cell in the cohort up to its current age equal to 0
	b = B; //set the original number of cells in the cohort equal to B
}

int cohort::integrate_cohort(int out, int out2) { 
	/* integrates the cohort until the age where F < Ft
	   if cohorts are created by division, they are added to the list cohnew (globally defined list of cohorts)
	   after each age integration step, we check if cells divided during this step (due to the function beta)
	          and whether the conditions are fulfilled that say that all the remaining cells in the cohort should divide
	          -> when the latter conditions are fulfilled, all remaining cells in the cohort divide
	   parameter out (int): = 1 (print full output), = 0 (print only the most important output) 
	   parameter out2 : used to determine if Matlab-file numberofcells.m has to be created 
			(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
			if == 0: no matlab-file is created, if == 1: the file is created*/

	realtype m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF; /*the values of m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF of the cells in the cohort at the current age integration output step*/
	realtype F_old, m_old, CycB_T_old, Cdh1_old, Cdc20_T_old, Cdc20_A_old, IEP_old, CKI_T_old, SK_old, TF_old; /*the values of F,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF of the cells in the cohort at the previous age integration output step*/
	realtype Fwdiv; /*value of F when there's only decrease in F due to death during an age integration step*/
	int Fto0 = 0; /*if equal to 1: the conditions are fulfilled that say that all the remaining cells in the cohort should divide*/

	/*code needed to use the CVODE solver for the age integration of the cohort*/
	realtype reltol = RELTOL;
	realtype abstol = ABSTOL;
	int flag;
	int iout = 0; //starting output time
	realtype t = iout; //current t (so age of the cohort)
	realtype tout = T1; //tout is the next output time (unless a root is found)
	void *cvode_mem;
	N_Vector y = N_VNew_Serial(NEQ);
	for(int i=0; i<NDIM; i++)
		NV_Ith_S(y,i) = init[i];
	NV_Ith_S(y,NDIM) = F;
	NV_Ith_S(y,NDIM+1) = theta;	

	ofstream myfile_out2;
	myfile_out2.precision(10);
	if(out2==1){
		myfile_out2.open("(m_F_numberofcells)foreverycohort.m", ios::app);
		myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; // the starting values of the cohort
	}

	if(out==1) //print the starting values of the cohort
		PrintOutput(iout,NV_Ith_S(y,0),NV_Ith_S(y,1),NV_Ith_S(y,2),NV_Ith_S(y,3),NV_Ith_S(y,4),NV_Ith_S(y,5),NV_Ith_S(y,6),NV_Ith_S(y,7),NV_Ith_S(y,8),NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
	realtype toutprev = 0; //save the previous output time in toutprev

	/*booleans vwm_min, vwm_max0 and vwm_max1 defined to handle the different cases 
	  for the conditions on m_min (if cond_min is 1: m has to be >= m_min for a cell to divide)
	  and m_max (if cond_max is 0: m_max is in equation for dm/dt and cells automatically can't grow larger than m_max
	                            stop integration when m<=(m_max*(1-eps_m)) to decrease the calculation time
				 if cond_max is 1: m_max is not in equation for dm/dt, so we explicitly assume that cells die 
				                (or no longer can divide) if m>m_max) */
	bool vwm_min = true;
	if(cond_min==1)
		vwm_min = (NV_Ith_S(y,0)>=m_min);
	bool vwm_max0 = true;
	bool vwm_max1 = true;
	if(cond_max==0)
		vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
	if(cond_max==1)
		vwm_max1 = (NV_Ith_S(y,0)<=m_max);
	
	if(((NV_Ith_S(y,1))>=(X_div-1.0e-6))                      //all cells should divide immediately when: 
			&& ((NV_Ith_S(y,1))<=(X_div+1.0e-6))              //  CycB_T in [X_div-1.0e-6,X_div+1.0e-6]
			&& ((k1-(k2p+k2pp*NV_Ith_S(y,2)+k2ppp*NV_Ith_S(y,4))*NV_Ith_S(y,1)) < 0)  //  and dCycB_T/dt<0
			&& vwm_min                                       //  and m>=m_min if cond_min is 1 (always true otherwise)
			&& vwm_max1){                                    //  and m<=m_max if cond_max is 1 (always true otherwise)
		if(out==1){
			myfile_OC << "Conditions for division are immediately fulfilled: all cells divide immediately" << endl;
			myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
		}
		m = NV_Ith_S(y,0);
		CycB_T = NV_Ith_S(y,1);
		Cdh1 = NV_Ith_S(y,2);
		Cdc20_T = NV_Ith_S(y,3);
		Cdc20_A = NV_Ith_S(y,4);
		IEP = NV_Ith_S(y,5);
		CKI_T = NV_Ith_S(y,6);
		SK = NV_Ith_S(y,7);
		TF = NV_Ith_S(y,8);
		theta = NV_Ith_S(y,NDIM+1); /*amount of nutrient consumed by one cell in the cohort up to its current age (when it divides)*/
		realtype in = NV_Ith_S(y,NDIM)*b; /*the number of cells that divide*/
		addcohort(phi,cohnew,in,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out); /*let the cells divide: add the resulting cohorts to the list cohnew*/
		if(out==1){
			myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
			printListCoh_OC(cohnew);
		}
	}
	else{ // when cells in the cohort don't all divide immediately
		while((Fto0!=1) //repeat age integration step on the cohort while: conditions for division of the remaining cells are not fulfilled                 
				&& vwm_max0                                 // and  m<=(m_max*(1-eps_m)) if cond_max is 0 (always true otherwise)
				&& vwm_max1                                 // and  m<=m_max if cond_max=1 (always true otherwise)
				&& (NV_Ith_S(y,NDIM)>=Ft)){                 // and  survival probability of the cells in the cohort >= Ft
			/*save the values of F,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF of the cells in the cohort at the previous age integration output step*/
			F_old = NV_Ith_S(y,NDIM);
			m_old = NV_Ith_S(y,0);
			CycB_T_old = NV_Ith_S(y,1);
			Cdh1_old = NV_Ith_S(y,2);
			Cdc20_T_old = NV_Ith_S(y,3);
			Cdc20_A_old = NV_Ith_S(y,4);
			IEP_old = NV_Ith_S(y,5);
			CKI_T_old = NV_Ith_S(y,6);
			SK_old = NV_Ith_S(y,7);
			TF_old = NV_Ith_S(y,8);
			
			if(toutprev==0){
				if((NV_Ith_S(y,1)>=(X_div-1.0e-6)) && (NV_Ith_S(y,1)<=(X_div+1.0e-6))){ 
					// if age integration starts in root CycB_T=X_div: take age integration step without RootFinding
					cvode_mem = CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
					if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);
					flag = CVodeInit(cvode_mem,f,T0,y);
					if (check_flag(&flag, "CVodeInit", 1)) return(1);
					flag = CVodeSStolerances(cvode_mem,reltol,abstol);
					if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);
					/*change the maximum number of steps to be taken by the solver in its attempt to reach the next output time from its default value 500*/
					flag = CVodeSetMaxNumSteps(cvode_mem, 10000); 
					if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);
					flag = CVode(cvode_mem,tout,y,&t,CV_NORMAL); //take age integration step using the CVODE solver

					if(out==1)
						PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,4), NV_Ith_S(y,5), NV_Ith_S(y,6), NV_Ith_S(y,7), NV_Ith_S(y,8), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
					if(out2==1)
						myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
					Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
																//ATTENTION: only valid when individual mortality function nu = D
					toutprev = t; //save current output time in toutprev 
					tout += TIN;  //next output time

					/*update the booleans*/
					if(cond_min==1)
						vwm_min = (NV_Ith_S(y,0)>=m_min);
					if(cond_max==0)
						vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
					if(cond_max==1)
						vwm_max1 = (NV_Ith_S(y,0)<=m_max);

					if(NV_Ith_S(y,NDIM)<(Fwdiv-1.0e-6)){ 
					//if F has changed more than only due to death (taking into account the computation precision) 
					// => cells divided during this age integration step
						if(out==1){
							myfile_OC << "Some or all cells in the cohort divided" << endl;
							myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
							myfile_OC << "(in the previous step " << F_old*b << " cells in the cohort were still alive)" << endl;
							myfile_OC << "(without division " << Fwdiv*b << " cells in the cohort would have been alive)" << endl;
							myfile_OC << RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b) << " cells have divided" << endl;
							myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
						}
						theta = NV_Ith_S(y,NDIM+1);
						realtype in = RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b); //the number of cells that divided 
						//let these cells (with state values m_old,CycB_T_old,...,TF_old) divide: add the resulting cohorts to the list cohnew
						addcohort(phi,cohnew,in,m_old,CycB_T_old,Cdh1_old,Cdc20_T_old,Cdc20_A_old,IEP_old,CKI_T_old,SK_old,TF_old,out); 
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}

					//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
					if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                        // if X in [X_div-1.0e-6,X_div+1.0e-6]
							&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
							&& ((k1-(k2p+k2pp*NV_Ith_S(y,2)+k2ppp*NV_Ith_S(y,4))*NV_Ith_S(y,1)) < 0)  // and dCycB_T/dt<0
							&& vwm_min                                            // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
							&& vwm_max1){                                         // and m<=m_max if cond_max=1 (always true otherwise)
						Fto0 = 1;
						if(out==1){
							myfile_OC << "All remaining cells in the cohort divide" << endl;
							myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive and divide (from the original " << b << " cells in the cohort)" << endl;
							myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
						}
						m = NV_Ith_S(y,0);
						CycB_T = NV_Ith_S(y,1);
						Cdh1 = NV_Ith_S(y,2);
						Cdc20_T = NV_Ith_S(y,3);
						Cdc20_A = NV_Ith_S(y,4);
						IEP = NV_Ith_S(y,5);
						CKI_T = NV_Ith_S(y,6);
						SK = NV_Ith_S(y,7);
						TF = NV_Ith_S(y,8);
						theta = NV_Ith_S(y,NDIM+1);
						realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
						addcohort(phi,cohnew,in,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out);
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}
					F_old = NV_Ith_S(y,NDIM);
					m_old = NV_Ith_S(y,0);
					CycB_T_old = NV_Ith_S(y,1);
					Cdh1_old = NV_Ith_S(y,2);
					Cdc20_T_old = NV_Ith_S(y,3);
					Cdc20_A_old = NV_Ith_S(y,4);
					IEP_old = NV_Ith_S(y,5);
					CKI_T_old = NV_Ith_S(y,6);
					SK_old = NV_Ith_S(y,7);
					TF_old = NV_Ith_S(y,8);

					CVodeFree(&cvode_mem);
				}

				if((Fto0!=1) //do second age integration step on the cohort if conditions for division of the remaining cells are not fulfilled                 
					&& vwm_max0                     // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
					&& vwm_max1                     // and m<=m_max if cond_max=1 (always true otherwise)
					&& (NV_Ith_S(y,NDIM)>=Ft)){     // and survival probability of the cells in the cohort >= Ft
					//initialize age integration with Rootfinding
					cvode_mem = CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
					if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);
					flag = CVodeInit(cvode_mem,f,toutprev,y);
					if (check_flag(&flag, "CVodeInit", 1)) return(1);
					flag = CVodeSStolerances(cvode_mem,reltol,abstol);
					if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);
					/*change the maximum number of steps to be taken by the solver in its attempt to reach the next output time from its default value 500*/
					flag = CVodeSetMaxNumSteps(cvode_mem, 10000); 
					if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);
				}
				
			}
			
			if((Fto0!=1) //do age integration step on the cohort if conditions for division of the remaining cells are not fulfilled                 
					&& vwm_max0                    // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
					&& vwm_max1                    // and m<=m_max if cond_max=1 (always true otherwise)
					&& (NV_Ith_S(y,NDIM)>=Ft)){    // and survival probability of the cells in the cohort >= Ft
				flag = CVodeRootInit(cvode_mem, 1, g);
				if (check_flag(&flag, "CVodeRootInit", 1)) return(1);
				flag = CVode(cvode_mem,tout,y,&t,CV_NORMAL); //take age integration step using the CVODE solver	

				if(out==1)
					PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,4),NV_Ith_S(y,5), NV_Ith_S(y,6), NV_Ith_S(y,7), NV_Ith_S(y,8), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
				if(out2==1)
					myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
				Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
															//ATTENTION: only valid when individual mortality function nu = D
				toutprev = t; //save current output time in toutprev 
				tout += TIN;   //next output time

				/*update the booleans*/
				if(cond_min==1)
					vwm_min = (NV_Ith_S(y,0)>=m_min);
				if(cond_max==0)
					vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
				if(cond_max==1)
					vwm_max1 = (NV_Ith_S(y,0)<=m_max);

				if(NV_Ith_S(y,NDIM)<(Fwdiv-1.0e-6)){ 
				//if F has changed more than only due to death (taking into account the computation precision) 
				// => cells divided during this age integration step
					if(out==1){
						myfile_OC << "Some or all cells in the cohort divided" << endl;
						myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
						myfile_OC << "(in the previous step " << F_old*b << " cells in the cohort were still alive)" << endl;
						myfile_OC << "(without division " << Fwdiv*b << " cells in the cohort would have been alive)" << endl;
						myfile_OC << RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b) << " cells have divided" << endl;
						myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
					}
					theta = NV_Ith_S(y,NDIM+1);
					realtype in = RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b); //the number of cells that divided 
					//let these cells (with state values m_old,CycB_T_old,...,TF_old) divide: add the resulting cohorts to the list cohnew
					addcohort(phi,cohnew,in,m_old,CycB_T_old,Cdh1_old,Cdc20_T_old,Cdc20_A_old,IEP_old,CKI_T_old,SK_old,TF_old,out); 
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}

				//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
				if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                       // if X in [X_div-1.0e-6,X_div+1.0e-6]
						&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
						&& ((k1-(k2p+k2pp*NV_Ith_S(y,2)+k2ppp*NV_Ith_S(y,4))*NV_Ith_S(y,1)) < 0)  // and dCycB_T/dt<0
						&& vwm_min                                          // and m>=m_min
						&& vwm_max0                                         // and m<=(m_max*(1-eps_m)) if cond_max is 0 (always true otherwise)
						&& vwm_max1){                                       // and m<=m_max if cond_max is 1 (always true otherwise)
					Fto0 = 1;
					if(out==1){
						myfile_OC << "All remaining cells in the cohort divide" << endl;
						myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive and divide (from the original " << b << " cells in the cohort)" << endl;
						myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
					}
					m = NV_Ith_S(y,0);
					CycB_T = NV_Ith_S(y,1);
					Cdh1 = NV_Ith_S(y,2);
					Cdc20_T = NV_Ith_S(y,3);
					Cdc20_A = NV_Ith_S(y,4);
					IEP = NV_Ith_S(y,5);
					CKI_T = NV_Ith_S(y,6);
					SK = NV_Ith_S(y,7);
					TF = NV_Ith_S(y,8);
					theta = NV_Ith_S(y,NDIM+1);
					realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
					addcohort(phi,cohnew,in,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out);
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}
			}
		}

		if((Fto0==1) && (out==1)) //if Fto0=1: all cells should have divided
			myfile_OC << "all cells should have divided " << endl; 
		if(Fto0!=1){ //if Fto0!=1: not all cells divided, but they can't divide/survive anymore (so we can stop the age integration) 
			if(NV_Ith_S(y,NDIM)<Ft){
				theta = NV_Ith_S(y,NDIM+1);
				if(out==1)
					myfile_OC << "Survival probability too small in cohort, cells in cohort do not divide (anymore)" << endl;
			}
			else if(!vwm_max0){
				theta = NV_Ith_S(y,NDIM+1);
				if(out==1)
					myfile_OC << "m too close to m_max, cells in cohort do not divide (anymore)" << endl;
			}
			else if(!vwm_max1){
				theta = NV_Ith_S(y,NDIM+1);
				if(out==1)
					myfile_OC << "m > m_max, cells in cohort do not divide (anymore)" << endl;
			}
			else
				myfile_OC << "Problem with age integration" << endl;
		}
		CVodeFree(&cvode_mem);
	}

	if(out2==1)
		myfile_out2.close(); 

	N_VDestroy_Serial(y);
	return(0);
}

void merge_2Coh(clist &cohnew, clistit &cohnewit, int out){
	/*merge 2 cohorts in the list of cohorts cohnew: the cohort at cohnewit and the one at cohnewit+1
	  weighted means are used for the values of m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK and TF
	  resulting list of cohorts in cohnew and pointer cohnewit to the resulting merged cohort
	  parameter out (int): = 1 (give full output), = 0 (give only important output) */

	if(out==1)
		myfile_OC << "merge two cohorts together" << endl;

	/*store the values of the left cohort in left_m,left_CycB_T,...,left_TF,left_b*/
	realtype left_m = cohnewit->getMass();
	realtype left_CycB_T = cohnewit->getCycB_T();
	realtype left_Cdh1 = cohnewit->getCdh1();
	realtype left_Cdc20_T = cohnewit->getCdc20_T();
	realtype left_Cdc20_A = cohnewit->getCdc20_A();
	realtype left_IEP = cohnewit->getIEP();
	realtype left_CKI_T = cohnewit->getCKI_T();
	realtype left_SK = cohnewit->getSK();
	realtype left_TF = cohnewit->getTF();
	realtype left_b = cohnewit->getb();
	if(out==1)
		myfile_OC << "merge left cohort: " << left_m << " " << left_CycB_T << " " << left_Cdh1 << " " << left_Cdc20_T << " " << left_Cdc20_A << " " << left_IEP << " " << left_CKI_T << " " << left_SK << " " << left_TF << " " << left_b << endl;

	cohnewit = cohnew.erase(cohnewit); //erase the left cohort, right cohort now at position cohnewit

	/*merge the cells of the left cohort with the cells in the right cohort at cohnewit*/
	realtype old_m = cohnewit->getMass();
	realtype old_CycB_T = cohnewit->getCycB_T();
	realtype old_Cdh1 = cohnewit->getCdh1();
	realtype old_Cdc20_T = cohnewit->getCdc20_T();
	realtype old_Cdc20_A = cohnewit->getCdc20_A();
	realtype old_IEP = cohnewit->getIEP();
	realtype old_CKI_T = cohnewit->getCKI_T();
	realtype old_SK = cohnewit->getSK();
	realtype old_TF = cohnewit->getTF();
	realtype old_b = cohnewit->getb();
	if(out==1)
		myfile_OC << "with right cohort: " << old_m << " " <<  old_CycB_T << " " << old_Cdh1 << " " << old_Cdc20_T << " " << old_Cdc20_A << " " << old_IEP << " " << old_CKI_T << " " << old_SK << " " << old_TF << " " << old_b << endl;
	realtype new_m = RCONST((old_b*old_m+left_b*left_m)/(old_b+left_b));
	realtype new_CycB_T = RCONST((old_b*old_CycB_T+left_b*left_CycB_T)/(old_b+left_b));
	realtype new_Cdh1 = RCONST((old_b*old_Cdh1+left_b*left_Cdh1)/(old_b+left_b));
	realtype new_Cdc20_T = RCONST((old_b*old_Cdc20_T+left_b*left_Cdc20_T)/(old_b+left_b));
	realtype new_Cdc20_A = RCONST((old_b*old_Cdc20_A+left_b*left_Cdc20_A)/(old_b+left_b));
	realtype new_IEP = RCONST((old_b*old_IEP+left_b*left_IEP)/(old_b+left_b));
	realtype new_CKI_T = RCONST((old_b*old_CKI_T+left_b*left_CKI_T)/(old_b+left_b));
	realtype new_SK = RCONST((old_b*old_SK+left_b*left_SK)/(old_b+left_b));
	realtype new_TF = RCONST((old_b*old_TF+left_b*left_TF)/(old_b+left_b));
	realtype new_b = RCONST(old_b+left_b);
	if(out==1)
		myfile_OC << "resulting cohort: " << new_m << " " << new_CycB_T << " " << new_Cdh1 << " " << new_Cdc20_T << " " << new_Cdc20_A << " " << new_IEP << " " << new_CKI_T << " " << new_SK << " " << new_TF << " " << new_b << endl;

	cohnewit->set_init(new_m,new_CycB_T,new_Cdh1,new_Cdc20_T,new_Cdc20_A,new_IEP,new_CKI_T,new_SK,new_TF,new_b); //change the values of the right cohort to these new "merged values"
}

void merge_Coh(clist &cohnew, clistit &cohnewit, realtype cell_m, realtype cell_CycB_T, realtype cell_Cdh1, realtype cell_Cdc20_T, realtype cell_Cdc20_A, realtype cell_IEP, realtype cell_CKI_T, realtype cell_SK, realtype cell_TF, realtype cell_b, int out){
	/*merge cell_b cells (with mass cell_m, cell_CycB_T, cell_Cdh1, cell_Cdc20_T, cell_Cdc20_A, cell_IEP, cell_CKI_T, cell_SK and cell_TF) with the cohort at location cohnewit in the list cohnew
	  weighted means are used for the values of m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK and TF
	  resulting list of cohorts in cohnew and pointer cohnewit to the resulting merged cohort
	  parameter out (int): =1 (give full output), = 0 (give only important output) */

	if(out==1)
		myfile_OC << "merge the cells: " << cell_m << " " << cell_CycB_T << " " << cell_Cdh1 << " " << cell_Cdc20_T << " " << cell_Cdc20_A << " " << cell_IEP << " " << cell_CKI_T << " " << cell_SK << " " << cell_TF  << " " << cell_b << endl;
	/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit (old_m,old_CycB_T,old_Cdh1,old_Cdc20_T,old_Cdc20_A,old_IEP,old_CKI_T,old_SK,old_TF,old_b)*/
	realtype old_m = cohnewit->getMass();
	realtype old_CycB_T = cohnewit->getCycB_T();
	realtype old_Cdh1 = cohnewit->getCdh1();
	realtype old_Cdc20_T = cohnewit->getCdc20_T();
	realtype old_Cdc20_A = cohnewit->getCdc20_A();
	realtype old_IEP = cohnewit->getIEP();
	realtype old_CKI_T = cohnewit->getCKI_T();
	realtype old_SK = cohnewit->getSK();
	realtype old_TF = cohnewit->getTF();
	realtype old_b = cohnewit->getb();
	if(out==1)
		myfile_OC << "with the cohort: " << old_m << " " <<  old_CycB_T << " " << old_Cdh1 << " " << old_Cdc20_T << " " << old_Cdc20_A << " " << old_IEP << " " << old_CKI_T << " " << old_SK << " " << old_TF << " " << old_b << endl;
	realtype new_m = RCONST((old_b*old_m+cell_b*cell_m)/(old_b+cell_b));
	realtype new_CycB_T = RCONST((old_b*old_CycB_T+cell_b*cell_CycB_T)/(old_b+cell_b));
	realtype new_Cdh1 = RCONST((old_b*old_Cdh1+cell_b*cell_Cdh1)/(old_b+cell_b));
	realtype new_Cdc20_T = RCONST((old_b*old_Cdc20_T+cell_b*cell_Cdc20_T)/(old_b+cell_b));
	realtype new_Cdc20_A = RCONST((old_b*old_Cdc20_A+cell_b*cell_Cdc20_A)/(old_b+cell_b));
	realtype new_IEP = RCONST((old_b*old_IEP+cell_b*cell_IEP)/(old_b+cell_b));
	realtype new_CKI_T = RCONST((old_b*old_CKI_T+cell_b*cell_CKI_T)/(old_b+cell_b));
	realtype new_SK = RCONST((old_b*old_SK+cell_b*cell_SK)/(old_b+cell_b));
	realtype new_TF = RCONST((old_b*old_TF+cell_b*cell_TF)/(old_b+cell_b));
	realtype new_b = RCONST(old_b+cell_b);
	if(out==1)
		myfile_OC << "resulting cohort: " << new_m << " " << new_CycB_T << " " << new_Cdh1 << " " << new_Cdc20_T << " " << new_Cdc20_A << " " << new_IEP << " " << new_CKI_T << " " << new_SK << " " << new_TF << " " << new_b << endl;
	cohnewit->set_init(new_m,new_CycB_T,new_Cdh1,new_Cdc20_T,new_Cdc20_A,new_IEP,new_CKI_T,new_SK,new_TF,new_b); //change the values of the cohort at cohnewit to these new "merged values"
}

void addcohort_sub(clist &cohnew, clistit &cohnewit, realtype cell_m, realtype cell_CycB_T, realtype cell_Cdh1, realtype cell_Cdc20_T, realtype cell_Cdc20_A, realtype cell_IEP, realtype cell_CKI_T, realtype cell_SK, realtype cell_TF, realtype cell_b, int out){
	/*adds cell_b cells (with cell_m, cell_CycB_T, ..., cell_SK and cell_TF) to a list of cohorts cohnew
	whether the cells are added to to a new cohort in cohnew, or if they are merged with existing cohorts 
	      (by using weighted means) depends on the parameter delta: if the distance in mass to an existing cohort >= delta,
		  a new cohort is added, if the distance in mass is < delta, the daughter cohort is merged with the existing one
	starts the search at location cohnewit
	parameter out (int): = 1 (give full output), = 0 (give only important output) */
	cohort a;
	while((cohnewit!=cohnew.end()) && (cell_m>=(cohnewit->getMass()+delta))){
		cohnewit++;
	}
	if(cohnewit==cohnew.end()){ //if there isn't such a cohort: add the cells in a new cohort at the end of the list
		a.set_init(cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b);
		cohnew.push_back(a);
		cohnewit = cohnew.end();
		cohnewit--; //important for search for the biggest parts (starts searching at cohnewit)
		if(out==1)
			myfile_OC << "-> bigger than the mass of every cohort in the list + delta, so create new cohort at the end of the list" << endl;
	}
	else{ /*means that: cell_m<(cohnewit->getMass()+delta) 
			-> check if cells are in inhibition zone of the cohort at cohnewit, so whether cell_m>(cohnewit->getMass()-delta) */
		if(cell_m>(cohnewit->getMass()-delta)){ //in inhibition zone of cohort at cohnewit
			cohnewit++; //check if it is also in inhibition zone of the next cohort
			if(cohnewit!=cohnew.end()){ //IF there is a next cohort!
				if(cell_m>(cohnewit->getMass()-delta)){ //also in inhibition of next cohort, check to which cohort it is the closest
					if(out==1)
						myfile_OC << "-> in 2 inhibition zones" << endl;
					realtype massa2 = cohnewit->getMass(); //mass of cells in the right cohort
					cohnewit--;
					realtype massa1 = cohnewit->getMass(); //mass of cells in the left cohort
					if(out==1)
						myfile_OC << "m of left and right cohort: " << massa1 << " " << massa2 << endl;
					if(abs(cell_m-massa1)>abs(cell_m-massa2)){ 
						//if mass-distance to right cohort smaller: add cells to right cohort (at location cohnewit+1)
						if(out==1)
							myfile_OC << "add cells to right cohort" << endl;
						cohnewit++;
						/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,CycB_T,...,TF*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out); 
						realtype new_m = cohnewit->getMass();
						//check if "new" right cohort is not too close to the left cohort (distance in m smaller than delta)
						cohnewit--;
						if((new_m-cohnewit->getMass())<delta){ //if mass-distance is smaller than delta: merge the two cohorts together
							if(out==1)
								myfile_OC << "changed right cohort too close to the left cohort: merge the two cohorts" << endl;
							merge_2Coh(cohnew,cohnewit,out);
						}
					}
					else{ //add cells to left cohort
						if(out==1)
							myfile_OC << "add cells to left cohort" << endl;
						/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,CycB_T,...,TF*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out); 
						realtype new_m = cohnewit->getMass();
						//check if "new" left cohort is not too close to the right cohort (distance in m smaller than delta)
						cohnewit++; //cohnewit at right cohort
						if((cohnewit->getMass()-new_m)<delta){ //if mass-distance is smaller than delta: merge the two cohorts together
							if(out==1)
								myfile_OC << "changed left cohort too close to the right cohort: merge the two cohorts" << endl;
							cohnewit--; //cohnewit at left cohort
							merge_2Coh(cohnew,cohnewit,out); 
							if(out==1)
								myfile_OC << cohnewit->getMass() << endl;
						}
					}
				}
				else{ //the cells are not in the inhibition zone of the next cohort in the list, so add the cells to the previous cohort
					if(out==1)
						myfile_OC << "-> cells are in the inhibition zone of one cohort: merge them with this cohort" << endl;
					cohnewit--;
					/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,CycB_T,...,TF*/
					merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out);
				}
			}
			else{ //the cells are in the inhibition zone of the last cohort in the list: add the cells to this cohort
				if(out==1)
					myfile_OC << "-> cells are in the inhibition zone of the last cohort in the list: merge the cells with this cohort" << endl;
				cohnewit--;
				/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the last cohort by using weighted means for m,CycB_T,...,TF*/
				merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out);
			}
		}
		else{ //add the cells in a new cohort
			if(out==1)
				myfile_OC << "-> the cells are not in the inhibition zone of a cohort of the list, so add a new cohort" << endl;
			a.set_init(cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b);
			cohnew.insert(cohnewit,a);
		}
	}
}

void addcohort(realtype p, clist &cohnew, realtype in, realtype m, realtype CycB_T, realtype Cdh1, realtype Cdc20_T, realtype Cdc20_A, realtype IEP, realtype CKI_T, realtype SK, realtype TF, int out){
	/*adds the daughter cells of "in" dividing cells (with m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF) to a list of cohorts cohnew
	p is the parameter for mass division: daughter cells with p*m and with (1-p)*m are created
	whether the two resulting daughter cohorts are simply added to cohnew, or if they are merged with existing cohorts 
	      (by using weighted means) depends on the parameter delta: if the distance in mass to an existing cohort >= delta,
		  a new cohort is added, if the distance in mass is < delta, the daughter cohort is merged with the existing one
	parameter out (int): = 1 (give full output), = 0 (give only important output) */
	
	clistit cohnewit;
	cohort a;

	totaalb += 2*in; //increase the total amount of cells born during one age integration by 2 times the number of dividing cells

    /*determine cohort to which smallest parts after division contribute (so with p*m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out,in)*/
	if(out==1)
		myfile_OC << "daughter cells with smallest mass" << endl;
	if(!cohnew.empty()){ //if cohnew is non-empty: search if there is cohort in cohnew for which p*m<(mass of cells in cohort + delta)
		cohnewit = cohnew.begin();
		addcohort_sub(cohnew,cohnewit,RCONST(p*m),CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in,out);
	}
	else{ //if the list of cohorts is empty: created new cohort with p*m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in
		if(out==1)
			myfile_OC << "the list of cohorts is empty -> create a new cohort with these cells" << endl;
		a.set_init(RCONST(p*m),CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in);
		cohnew.push_back(a);
		cohnewit = cohnew.begin(); 
	}

	/*determine cohort to which biggest parts after division contribute (so with (1-p)*m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in)*/
	if(out==1)
		myfile_OC << "daughter cells with biggest mass" << endl;
	//start search at cohnewit where the small parts are inserted
	addcohort_sub(cohnew,cohnewit,RCONST((1-p)*m),CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in,out);
					
}

void integrate_allcohorts(clist coh, int out, int out2){ 
	/* corresponds to map M in paper 
	integrates a list of cohorts coh and calculates the adjustment to the nutrient 
	new birth cohorts in list cohnew, new S in tempS
	starting value of S is defined globally
	parameter out (int): =1 (give full output), = 0 (give only important output) 
	parameter out2 : used to determine if matlab-file numberofcells.m has to be created 
		(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
		if == 0: no matlab-file is created, if == 1: the file is created*/
	
	realtype con = RCONST(0); //total amount of nutrient consumed by the cells in the cohorts in coh during 1 age integration
	totaalb = RCONST(0); /*initialise the total amount of cells born during the age integration of coh 
	                      (calculated through the cells that divide)
	                       totaalb is defined globally*/
	totalb = RCONST(0); /*initialize the total amount of cells born during one age integration of coh 
						 (calculated through cohnew)
	                     totalb is defined globally*/
	/*empty cohnew*/
	clistit cohit;
	cohit = cohnew.begin();
	while(cohit!=cohnew.end())
		cohit = cohnew.erase(cohit);

	if(out==1){
		myfile_OC << "number of cohorts: " << (int) coh.size() << endl;
		myfile_OC << "S = " << S << endl;
		myfile_OC << "S0 = " << S0 << endl;
		myfile_OC << "D = " << D << endl;
		myfile_OC << endl;
	}

	ofstream myfile_out2;
	myfile_out2.precision(10);
	if(out2==1)
		myfile_out2.open("(m_F_numberofcells)foreverycohort.m", ios::app);

	int p = 1; //number of the cohort in list coh
	for(cohit=coh.begin();cohit!=coh.end();++cohit){ //integrates the given list of cohorts coh
		if(out==1)
			myfile_OC << "cohort " << p << " with initial mass " << cohit->getMass() << endl;
		if(out2==1){
			myfile_out2 << "coh" << p << " = [";
			myfile_out2.close();
		}
		if(cohit->getb()>0){ //integrate the cohort IF it is non-empty
			cohit->integrate_cohort(out,out2); 
			con += cohit->getb()*cohit->getTheta(); //calculate the amount of nutrient consumed by the cells in the cohort and adds it to con 
		}
		else{
			if(out==1)
				myfile_OC << "no cells in this cohort" << endl;
		}
		if(out2==1){
			myfile_out2.open("(m_F_numberofcells)foreverycohort.m", ios::app);
			myfile_out2 << "]" << endl;
		}
		p++;
	}

	if(out2==1)
		myfile_out2.close();


	realtype eps = RCONST(eps_b*0.01*totaalb); //calculate eps = the minimum amount of cells in a cohort so that the cohort can't be neglected
	                                           // = eps_b% of total amount of cells born
	if(out==1){
		myfile_OC << "total amount of cells born = " << totaalb << endl;
		myfile_OC << "epsilon = " << eps << endl;
	}

	/*loop over the list of created cohorts (cohnew): remove cohorts with b<eps and calculate the total amount of cells in the remaining cohorts (totalb)*/
	p = 1;
	if(out==1)
		myfile_OC << "overview of new cohorts" << endl;
	cohit = cohnew.begin();
	while(cohit!=cohnew.end()){
		if(cohit->getb()>=eps){
			totalb += cohit->getb();
			if(out==1)
				myfile_OC << "cohort " << p << " with initial mass " << cohit->getMass() << " CycB_T " << cohit->getCycB_T() << " Cdh1 " << cohit->getCdh1() << " Cdc20_T " << cohit->getCdc20_T() << " Cdc20_A " << cohit->getCdc20_A() << " IEP " << cohit->getIEP() << " CKI_T " << cohit->getCKI_T() << " SK " << cohit->getSK() << " TF " << cohit->getTF() << " number " << cohit->getb() << endl;
			p++;
			cohit++;
		}
		else{
			if(out==1)
				myfile_OC << "cohort erased with initial mass " << cohit->getMass() << " because amount of cells smaller than epsilon (" << cohit->getb() << ")" << endl;
			cohit = cohnew.erase(cohit);
		}
	}

	tempS = RCONST(S+prop*(RCONST(S0-(1.0/D)*con)-S)); //calculate new S for the next age integration

	if(out==1){
		myfile_OC << "total births in all cohorts: " << totalb << endl;
		myfile_OC << "total amount of nutrient used by all the cohorts: " << con << endl;
		myfile_OC << "intrinsic change of nutrient per unit of time " << D*(S0-S) << endl;
		myfile_OC << "nutrient concentration for next age integration (unadjusted) " << RCONST(S0-(1.0/D)*con) << endl;
		myfile_OC << "nutrient concentration for next age integration (adjusted) " << RCONST(S+prop*(RCONST(S0-(1.0/D)*con)-S)) << endl;
		if(RCONST(S+prop*(RCONST(S0-(1.0/D)*con)-S))<0.000001)
			myfile_OC << "nutrient concentration for next age integration (adjusted) too small, so set S equal to 0.000001" << endl;
		myfile_OC << endl;
	}
}

void addcohort_dest_sub(clist &cohnew, clistit &cohnewit, int &i_cells, int i_start, int &how, realtype cell_m, realtype cell_CycB_T, realtype cell_Cdh1, realtype cell_Cdc20_T, realtype cell_Cdc20_A, realtype cell_IEP, realtype cell_CKI_T, realtype cell_SK, realtype cell_TF, realtype cell_b, int out){
	/*adds cell_b cells (with cell_m, cell_CycB_T, ..., cell_SK and cell_TF) to a list of cohorts cohnew
	whether the cells are added to to a new cohort in cohnew, or if they are merged with existing cohorts 
	      (by using weighted means) depends on the parameter delta: if the distance in mass to an existing cohort >= delta,
		  a new cohort is added, if the distance in mass is < delta, the daughter cohort is merged with the existing one
	starts the search at location cohnewit
	see global definition for i_cells (starts at i_start) and for how -> used for i_sm or i_gr and for how_sm or how_gr
	parameter out (int): = 1 (give full output), = 0 (give only important output) */
	cohort a;
	i_cells = i_start;
	while((cohnewit!=cohnew.end()) && (cell_m>=(cohnewit->getMass()+delta))){
		cohnewit++;
		i_cells++;
	}
	if(cohnewit==cohnew.end()){ //if there isn't such a cohort: add the cells in a new cohort at the end of the list
		a.set_init(cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b);
		cohnew.push_back(a);
		cohnewit = cohnew.end();
		cohnewit--; //important for search for the biggest parts (starts searching at cohnewit)
		if(out==1)
			myfile_OC << "-> bigger than the mass of every cohort in the list + delta, so create new cohort at the end of the list" << endl;
		how = 1; 
		i_cells--; 
	}
	else{ /*means that: cell_m<(cohnewit->getMass()+delta) 
			-> check if cells are in inhibition zone of the cohort at cohnewit, so whether cell_m>(cohnewit->getMass()-delta) */
		if(cell_m>(cohnewit->getMass()-delta)){ //in inhibition zone of cohort at cohnewit
			cohnewit++; //check if it is also in inhibition zone of the next cohort
			if(cohnewit!=cohnew.end()){ //IF there is a next cohort
				if(cell_m>(cohnewit->getMass()-delta)){ //also in inhibition of next cohort, check to which cohort it is the closest
					if(out==1)
						myfile_OC << "-> in 2 inhibition zones" << endl;
					realtype massa2 = cohnewit->getMass(); //mass of cells in the right cohort
					cohnewit--;
					realtype massa1 = cohnewit->getMass(); //mass of cells in the left cohort
					if(out==1)
						myfile_OC << "m of left and right cohort: " << massa1 << " " << massa2 << endl;
					if(abs(cell_m-massa1)>abs(cell_m-massa2)){ 
						//if mass-distance to right cohort smaller: add cells to right cohort (at location cohnewit+1)
						if(out==1)
							myfile_OC << "add cells to right cohort" << endl;
						cohnewit++;
						/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out); 
						how = 2;
						i_cells++;
						realtype new_m = cohnewit->getMass();
						//check if "new" right cohort is not too close to the left cohort (distance in m smaller than delta)
						cohnewit--;
						if((new_m-cohnewit->getMass())<delta){ //if mass-distance is smaller than delta: merge the two cohorts together
							if(out==1)
								myfile_OC << "changed right cohort too close to the left cohort: merge the two cohorts" << endl;
							merge_2Coh(cohnew,cohnewit,out);
							how = 3;
							i_cells--;
						}
					}
					else{ //add cells to left cohort
						if(out==1)
							myfile_OC << "add cells to left cohort" << endl;
						/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out); 
						how = 2;
						realtype new_m = cohnewit->getMass();
						//check if "new" left cohort is not too close to the right cohort (distance in m smaller than delta)
						cohnewit++; //cohnewit at right cohort
						if((cohnewit->getMass()-new_m)<delta){ //if mass-distance is smaller than delta: merge the two cohorts together
							if(out==1)
								myfile_OC << "changed left cohort too close to the right cohort: merge the two cohorts" << endl;
							cohnewit--; //cohnewit at left cohort
							merge_2Coh(cohnew,cohnewit,out); 
							how = 3;
							if(out==1)
								myfile_OC << cohnewit->getMass() << endl;
						}
						cohnewit--;
					}
				}
				else{ //the cells are not in the inhibition zone of the next cohort in the list, so add the cells to the previous cohort
					if(out==1)
						myfile_OC << "-> cells are in the inhibition zone of one cohort: merge them with this cohort" << endl;
					cohnewit--;
					/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF*/
					merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out);
					how = 2;
				}
			}
			else{ //the cells are in the inhibition zone of the last cohort in the list: add the cells to this cohort
				if(out==1)
					myfile_OC << "-> cells are in the inhibition zone of the last cohort in the list: merge the cells with this cohort" << endl;
				cohnewit--;
				/*merge the new cells (cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b) with the cells in the last cohort by using weighted means for m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF*/
				merge_Coh(cohnew,cohnewit,cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b,out);
				how = 2;
			}
		}
		else{ //add the cells in a new cohort
			if(out==1)
				myfile_OC << "-> the cells are not in the inhibition zone of a cohort of the list, so add a new cohort" << endl;
			a.set_init(cell_m,cell_CycB_T,cell_Cdh1,cell_Cdc20_T,cell_Cdc20_A,cell_IEP,cell_CKI_T,cell_SK,cell_TF,cell_b);
			cohnew.insert(cohnewit,a);
			cohnewit--;
			how = 1;
			i_cells--;
		}
	}
}

void addcohort_dest(realtype p, clist &cohnew, realtype in, realtype m, realtype CycB_T, realtype Cdh1, realtype Cdc20_T, realtype Cdc20_A, realtype IEP, realtype CKI_T, realtype SK, realtype TF, int out, int &i_sm, int &how_sm, int &i_gr, int &how_gr){
	/*adds the daughter cells of "in" dividing cells (with m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK and TF) to a list of cohorts cohnew
	p is the parameter for mass division: daughter cells with p*m and with (1-p)*m are created
	whether the two resulting daughter cohorts are simply added to cohnew, or if they are merged with existing cohorts 
	      (by using weighted means) depends on the parameter delta: if the distance in mass to an existing cohort >= delta,
		  a new cohort is added, if the distance in mass is < delta, the daughter cohort is merged with the existing one
	see global definition for i_sm, how_sm, i_gr and how_gr -> contains the information needed to print data_figd3_fp.txt in integrate_allcohorts_dest
	parameter out (int): = 1 (give full output), = 0 (give only important output) */
	
	clistit cohnewit;
	cohort a;

	totaalb += 2*in; //increase the total amount of cells born during one age integration by 2 times the number of dividing cells

    /*determine cohort to which smallest parts after division contribute (so with p*m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in)*/
	if(out==1)
		myfile_OC << "daughter cells with smallest mass" << endl;
	if(!cohnew.empty()){ //if cohnew is non-empty: search if there is cohort in cohnew for which p*m<(mass of cells in cohort + delta)
		cohnewit = cohnew.begin();
		addcohort_dest_sub(cohnew,cohnewit,i_sm,1,how_sm,RCONST(p*m),CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in,out);
	}
	else{ //if the list of cohorts is empty: created new cohort with p*m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in
		if(out==1)
			myfile_OC << "the list of cohorts is empty -> create a new cohort with these cells" << endl;
		a.set_init(RCONST(p*m),CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in);
		cohnew.push_back(a);
		cohnewit = cohnew.begin(); 
		how_sm = 1;
		i_sm = 0;
	}

	int i_start; //cohort after the cohort where the small daughter cells are and where the search for the insertion of the big daughter cells has to start
	if(how_sm==1)
		i_start = i_sm+1;
	else
		i_start = i_sm;

	/*determine cohort to which biggest parts after division contribute (so with (1-p)*m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in)*/
	if(out==1)
		myfile_OC << "daughter cells with biggest mass"<< endl;
	//start search at cohnewit where the small parts are inserted
	addcohort_dest_sub(cohnew,cohnewit,i_gr,i_start,how_gr,RCONST((1-p)*m),CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,in,out);
}

int cohort::integrate_cohort_dest(int out, int out2, int &i_sm, int &how_sm, int &i_gr, int &how_gr) { 
	/* integrates the cohort until the age where F < Ft
	   prints data_figd3_fp.txt with the information needed for the cohort-to-cohort representation
	   if cohorts are created by division, they are added to the list cohnew (globally defined list)
	   after each age integration step, we check if cells divided during this step (due to the function beta)
	          and whether the conditions are fulfilled that say that all the cells in the cohort should be divided
	   when these conditions are fulfilled, all remaining cells in the cohort divide
	   parameter out (int): = 1 (gives full output), = 0 (gives only the most important output) 
	   parameter out2 : used to determine if matlab-file numberofcells.m has to be created 
			(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
			if == 0: no matlab-file is created, if == 1: the file is created*/

	realtype m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF; /*the internal state values of the cells in the cohort at the current age integration output step*/
	realtype F_old, m_old, CycB_T_old, Cdh1_old, Cdc20_T_old, Cdc20_A_old, IEP_old, CKI_T_old, SK_old, TF_old; /*the values of F,m,CycB_T,...,TF of the cells in the cohort at the previous age integration output step*/
	realtype Fwdiv; /*value of F when there's only decrease in F due to death during an age integration step*/
	int Fto0 = 0; /*if Fto0 = 1: the conditions are fulfilled that all the cells in the cohort should be divided*/

	/*code needed to use the CVODE solver for the age integration of the cohort*/
	realtype reltol = RELTOL;
	realtype abstol = ABSTOL;
	int flag;
	int iout = 0; //starting output time
	realtype t = iout; //current t (so age of the cohort)
	realtype tout = T1; //tout is the next output time (unless a root is found)
	void *cvode_mem;
	N_Vector y = N_VNew_Serial(NEQ);
	for (int i=0; i<NDIM; i++) {
		NV_Ith_S(y,i) = init[i];
	}
	NV_Ith_S(y,NDIM) = F;
	NV_Ith_S(y,NDIM+1) = theta;	

	ofstream myfile_out2;
	myfile_out2.precision(10);
	if(out2==1){
		myfile_out2.open("(m_F_numberofcells)foreverycohort.m", ios::app);
		myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; // the starting values of the cohort
		//the starting values of the cohort
	}

	if(out==1) //print the starting values of the cohort
		PrintOutput(iout,NV_Ith_S(y,0),NV_Ith_S(y,1),NV_Ith_S(y,2),NV_Ith_S(y,3),NV_Ith_S(y,4),NV_Ith_S(y,5),NV_Ith_S(y,6),NV_Ith_S(y,7),NV_Ith_S(y,8),NV_Ith_S(y,NDIM),NV_Ith_S(y,NDIM+1));
	realtype toutprev = 0; //save the previous output time in toutprev

	/*booleans vwm_min, vwm_max0 and vwm_max1 defined to handle the different cases 
	  for the conditions on m_min (if cond_min=1: m>=m_min for division)
	  and m_max (if cond_max=0: m_max in equation for dm/dt and cells automatically can't grow larger than m_max
	                            stop integration when m<=(m_max*(1-eps_m)) to decrease the calculation time
				 if cond_max=1:	m_max not in equation for dm/dt, so we explicitly assume that cells die 
				                (or no longer can divide) if m>m_max) 			*/
	bool vwm_min = true;
	if(cond_min==1)
		vwm_min = (NV_Ith_S(y,0)>=m_min);
	bool vwm_max0 = true;
	bool vwm_max1 = true;
	if(cond_max==0)
		vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
	if(cond_max==1)
		vwm_max1 = (NV_Ith_S(y,0)<=m_max);
	
	if(((NV_Ith_S(y,1))>=(X_div-1.0e-6))                      //all cells should divide immediately when: 
			&& ((NV_Ith_S(y,1))<=(X_div+1.0e-6))              //  X in [X_div-1.0e-6,X_div+1.0e-6]
			&& ((k1-(k2p+k2pp*NV_Ith_S(y,2)+k2ppp*NV_Ith_S(y,4))*NV_Ith_S(y,1)) < 0)   //and dCycB_T/dt<0
			&& vwm_min                                       //  and m>=m_min if cond_min=1 (always true otherwise)
			&& vwm_max1){                                    //  and m<=m_max if cond_max=1 (always true otherwise)
		if(out==1){
			myfile_OC << "Conditions for division are immediately fulfilled: all cells divide immediately" << endl;
			myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
		}
		m = NV_Ith_S(y,0);
		CycB_T = NV_Ith_S(y,1);
		Cdh1 = NV_Ith_S(y,2);
		Cdc20_T = NV_Ith_S(y,3);
		Cdc20_A = NV_Ith_S(y,4);
		IEP = NV_Ith_S(y,5);
		CKI_T = NV_Ith_S(y,6);
		SK = NV_Ith_S(y,7);
		TF = NV_Ith_S(y,8);
		theta = NV_Ith_S(y,NDIM+1); /*amount of nutrient consumed by one cell in the cohort up to its current age (when it divides)*/
		realtype in = NV_Ith_S(y,NDIM)*b; /*the number of cells (or concentration) that divide*/
		addcohort_dest(phi,cohnew,in,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out,i_sm,how_sm,i_gr,how_gr); /*let the cells divide: add the resulting cohorts to the list cohnew*/
		if(out==1){
			myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
			printListCoh_OC(cohnew);
		}
	}
	else{ // when cells in the cohort don't all divide immediately
		while((Fto0!=1) //repeat age integration step on the cohort while: conditions for division of the remaining cells are not fulfilled                 
				&& vwm_max0                                  //  and  m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
				&& vwm_max1                                 //  and  m<=m_max if cond_max=1 (always true otherwise)
				&& (NV_Ith_S(y,NDIM)>=Ft)){                 // and  survival probability of the cells in the cohort >= Ft
			/*save the values of F,m,CycB_T,...,TF of the cells in the cohort at the previous age integration output step*/
			F_old = NV_Ith_S(y,NDIM);
			m_old = NV_Ith_S(y,0);
			CycB_T_old = NV_Ith_S(y,1);
			Cdh1_old = NV_Ith_S(y,2);
			Cdc20_T_old = NV_Ith_S(y,3);
			Cdc20_A_old = NV_Ith_S(y,4);
			IEP_old = NV_Ith_S(y,5);
			CKI_T_old = NV_Ith_S(y,6);
			SK_old = NV_Ith_S(y,7);
			TF_old = NV_Ith_S(y,8);
			
			if(toutprev==0){
				if((NV_Ith_S(y,1)>=(X_div-1.0e-6)) && (NV_Ith_S(y,1)<=(X_div+1.0e-6))){ 
					// if age integration starts in root X=X_div: take age integration step without RootFinding
					cvode_mem = CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
					if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);
					flag = CVodeInit(cvode_mem,f,T0,y);
					if (check_flag(&flag, "CVodeInit", 1)) return(1);
					flag = CVodeSStolerances(cvode_mem,reltol,abstol);
					if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);
					/*change the maximum number of steps to be taken by the solver in its attempt to reach the next output time from its default value 500*/
					flag = CVodeSetMaxNumSteps(cvode_mem, 10000); 
					if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);
					flag = CVode(cvode_mem,tout,y,&t,CV_NORMAL); //take age integration step using the CVODE solver

					if(out==1)
						PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1),NV_Ith_S(y,2),NV_Ith_S(y,3),NV_Ith_S(y,4),NV_Ith_S(y,5),NV_Ith_S(y,6),NV_Ith_S(y,7),NV_Ith_S(y,8),NV_Ith_S(y,NDIM),NV_Ith_S(y,NDIM+1));
					if(out2==1)
						myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
					Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
																//ATTENTION: only valid when individual mortality function nu = D 
					toutprev = t; //save current output time in toutprev 
					tout+=TIN;   //next output time

					/*update the booleans*/
					if(cond_min==1)
						vwm_min = (NV_Ith_S(y,0)>=m_min);
					if(cond_max==0)
						vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
					if(cond_max==1)
						vwm_max1 = (NV_Ith_S(y,0)<=m_max);

					if(NV_Ith_S(y,NDIM)<(Fwdiv-1.0e-6)){ 
					//if F has changed more than only due to death (taking into account the computation precision) 
					// => cells divided during this age integration step
						if(out==1){
							myfile_OC << "Some or all cells in the cohort divided"<< endl;
							myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
							myfile_OC << "(in the previous step " << F_old*b << " cells in the cohort were still alive)" << endl;
							myfile_OC << "(without division " << Fwdiv*b << " cells in the cohort would have been alive)" << endl;
							myfile_OC << RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b) << " cells have divided" << endl;
							myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
						}
						theta = NV_Ith_S(y,NDIM+1);
						realtype in = RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b); //the number of cells that divided 
						//let these cells (with state values m_old,CycB_T_old,...,TF_old) divide: add the resulting cohorts to the list cohnew
						addcohort_dest(phi,cohnew,in,m_old,CycB_T_old,Cdh1_old,Cdc20_T_old,Cdc20_A_old,IEP_old,CKI_T_old,SK_old,TF_old,out,i_sm,how_sm,i_gr,how_gr); 
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}

					//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
					if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                        // if X in [X_div-1.0e-6,X_div+1.0e-6]
							&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
							&& ((k1-(k2p+k2pp*NV_Ith_S(y,2)+k2ppp*NV_Ith_S(y,4))*NV_Ith_S(y,1)) < 0)    // and dCycB_T/dt<0
							&& vwm_min                                            // and m>=m_min
							&& vwm_max0                                           // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
							&& vwm_max1){                                         // and m<=m_max if cond_max=1 (always true otherwise)
						Fto0 = 1;
						if(out==1){
							myfile_OC << "All remaining cells in the cohort divide" << endl;
							myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive and divide (from the original " << b << " cells in the cohort)" << endl;
							myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
						}
						m = NV_Ith_S(y,0);
						CycB_T = NV_Ith_S(y,1);
						Cdh1 = NV_Ith_S(y,2);
						Cdc20_T = NV_Ith_S(y,3);
						Cdc20_A = NV_Ith_S(y,4);
						IEP = NV_Ith_S(y,5);
						CKI_T = NV_Ith_S(y,6);
						SK = NV_Ith_S(y,7);
						TF = NV_Ith_S(y,8);
						theta = NV_Ith_S(y,NDIM+1);
						realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
						addcohort_dest(phi,cohnew,in,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out,i_sm,how_sm,i_gr,how_gr);
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}
					F_old = NV_Ith_S(y,NDIM);
					m_old = NV_Ith_S(y,0);
					CycB_T_old = NV_Ith_S(y,1);
					Cdh1_old = NV_Ith_S(y,2);
					Cdc20_T_old = NV_Ith_S(y,3);
					Cdc20_A_old = NV_Ith_S(y,4);
					IEP_old = NV_Ith_S(y,5);
					CKI_T_old = NV_Ith_S(y,6);
					SK_old = NV_Ith_S(y,7);
					TF_old = NV_Ith_S(y,8);

					CVodeFree(&cvode_mem);
				}

				if((Fto0!=1) //do second age integration step on the cohort if conditions for division of the remaining cells are not fulfilled                 
					&& vwm_max0                                  //  and  m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
					&& vwm_max1                                 //  and  m<=m_max if cond_max=1 (always true otherwise)
					&& (NV_Ith_S(y,NDIM)>=Ft)){                 // and  survival probability of the cells in the cohort >= Ft
					//initialize age integration with Rootfinding
					cvode_mem = CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
					if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);
					flag = CVodeInit(cvode_mem,f,toutprev,y);
					if (check_flag(&flag, "CVodeInit", 1)) return(1);
					flag = CVodeSStolerances(cvode_mem,reltol,abstol);
					if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);
					/*change the maximum number of steps to be taken by the solver in its attempt to reach the next output time from its default value 500*/
					flag = CVodeSetMaxNumSteps(cvode_mem, 10000); 
					if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);
				}
				
			}
			
			if((Fto0!=1) //do age integration step on the cohort if conditions for division of the remaining cells are not fulfilled                 
					&& vwm_max0                                  //  and  m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
					&& vwm_max1                                 //  and  m<=m_max if cond_max=1 ((always true otherwise)
					&& (NV_Ith_S(y,NDIM)>=Ft)){                 // and  survival probability of the cells in the cohort >= Ft
				flag = CVodeRootInit(cvode_mem, 1, g);
				if (check_flag(&flag, "CVodeRootInit", 1)) return(1);
				flag = CVode(cvode_mem,tout,y,&t,CV_NORMAL); //take age integration step using the CVODE solver	

				if(out==1)
					PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1),NV_Ith_S(y,2),NV_Ith_S(y,3),NV_Ith_S(y,4),NV_Ith_S(y,5),NV_Ith_S(y,6),NV_Ith_S(y,7),NV_Ith_S(y,8),NV_Ith_S(y,NDIM),NV_Ith_S(y,NDIM+1));
				if(out2==1)
					myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
				Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
															//ATTENTION: only valid when individual mortality function nu = D
				toutprev = t; //save current output time in toutprev 
				tout += TIN;   //next output time

				/*update the booleans*/
				if(cond_min==1)
					vwm_min = (NV_Ith_S(y,0)>=m_min);
				if(cond_max==0)
					vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
				if(cond_max==1)
					vwm_max1 = (NV_Ith_S(y,0)<=m_max);

				if(NV_Ith_S(y,NDIM)<(Fwdiv-1.0e-6)){ 
				//if F has changed more than only due to death (taking into account the computation precision) 
				// => cells divided during this age integration step
					if(out==1){
						myfile_OC << "Some or all cells in the cohort divided" << endl;
						myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
						myfile_OC << "(in the previous step " << F_old*b << " cells in the cohort were still alive)" << endl;
						myfile_OC << "(without division " << Fwdiv*b << " cells in the cohort would have been alive)" << endl;
						myfile_OC << RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b) << " cells have divided" << endl;
						myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
					}
					theta = NV_Ith_S(y,NDIM+1);
					realtype in = RCONST((Fwdiv-NV_Ith_S(y,NDIM))*b); //the number of cells that divided 
					//let these cells (with state values m_old,CycB_T_old,...,TF_old) divide: add the resulting cohorts to the list cohnew
					addcohort_dest(phi,cohnew,in,m_old,CycB_T_old,Cdh1_old,Cdc20_T_old,Cdc20_A_old,IEP_old,CKI_T_old,SK_old,TF_old,out,i_sm,how_sm,i_gr,how_gr); 
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}

				//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
				if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                      // if X in [X_div-1.0e-6,X_div+1.0e-6]
						&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
						&& ((k1-(k2p+k2pp*NV_Ith_S(y,2)+k2ppp*NV_Ith_S(y,4))*NV_Ith_S(y,1)) < 0)   // and dCycB_T/dt<0
						&& vwm_min                                          // and m>=m_min
						&& vwm_max0                                         // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
						&& vwm_max1){                                       // and m<=m_max if cond_max=1 (always true otherwise)
					Fto0 = 1;
					if(out==1){
						myfile_OC << "All remaining cells in the cohort divide" << endl;
						myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive and divide (from the original " << b << " cells in the cohort)" << endl;
						myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
					}
					m = NV_Ith_S(y,0);
					CycB_T = NV_Ith_S(y,1);
					Cdh1 = NV_Ith_S(y,2);
					Cdc20_T = NV_Ith_S(y,3);
					Cdc20_A = NV_Ith_S(y,4);
					IEP = NV_Ith_S(y,5);
					CKI_T = NV_Ith_S(y,6);
					SK = NV_Ith_S(y,7);
					TF = NV_Ith_S(y,8);
					theta = NV_Ith_S(y,NDIM+1);
					realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
					addcohort_dest(phi,cohnew,in,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,out,i_sm,how_sm,i_gr,how_gr);
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}
			}
		}

		if((Fto0==1) && (out==1)) //if Fto0=1: all cells should have divided
			myfile_OC << "all cells should have divided " << endl; 
		if(Fto0!=1){ //if Fto0!=1: not all cells divided, but they can't divide/survive anymore (so we can stop the age integration) 
			if(NV_Ith_S(y,NDIM)<Ft){
				theta = NV_Ith_S(y,NDIM+1);
				if(out==1)
					myfile_OC << "Survival probability too small in cohort, cells in cohort do not divide (anymore)" << endl;
				how_sm = -1;
				how_gr = -1;
			}
			else if(!vwm_max0){
				theta = NV_Ith_S(y,NDIM+1);
				if(out==1)
					myfile_OC << "m too close to m_max, cells in cohort do not divide (anymore)" << endl;
				how_sm = -1;
				how_gr = -1;
			}
			else if(!vwm_max1){
				theta = NV_Ith_S(y,NDIM+1);
				if(out==1)
					myfile_OC << "m > m_max, cells in cohort do not divide (anymore)" << endl;
				how_sm = -1;
				how_gr = -1;
			}
			else
				myfile_OC << "Problem with age integration" << endl;
		}
		CVodeFree(&cvode_mem);
	}

	if(out2==1)
		myfile_out2.close(); 

	N_VDestroy_Serial(y);
	return(0);
}

void integrate_allcohorts_dest(clist coh, int out, int out2){ 
	/* corresponds to map M in paper 
	integrates a list of cohorts coh and calculates the adjustment to the nutrient
	prints data_figd3_fp.txt (with the information needed for the cohort-to-cohort representation) with for every cohort of coh: 
	     mass, mass of the cohort of the small daughter cells, mass of the cohort of the large daughter cells, number of cells originally in the cohort
	new birth cohorts in list cohnew, new S in tempS
	starting value of S is defined globally
	parameter out (int): =1 (give full output), = 0 (give only important output) 
	parameter out2 : used to determine if matlab-file numberofcells.m has to be created 
		(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
		if == 0: no matlab-file is created, if == 1: the file is created*/
	
	realtype con = RCONST(0); //total amount of nutrient consumed by the cells in the cohorts in coh during 1 age integration
	totaalb = RCONST(0); /*initialise the total amount of cells born during one age integration of coh 
						 (calculated through the cells that divide)
	                     totaalb defined globally*/
	totalb = RCONST(0); /*initialise the total amount of cells born during one age integration of coh 
						(calculated through cohnew)
	                    totalb defined globally*/
	clistit cohit;

	alglib::real_1d_array destination_sm; //vector with for every one of the cohorts the number of the cohort of the small daughter cells
	alglib::real_1d_array destination_gr; //vector with for every one of the cohorts the number of the cohort of the big daughter cells
	destination_sm.setlength(coh.size()); 
	destination_gr.setlength(coh.size());

	/*empty cohnew*/
	cohit = cohnew.begin();
	while(cohit!=cohnew.end())
		cohit = cohnew.erase(cohit);

	if(out==1){
		myfile_OC << "number of cohorts: " << (int) coh.size() <<endl;
		myfile_OC << "S = " << S << endl;
		myfile_OC << "S0 = " << S0 << endl;
		myfile_OC << "D = " << D << endl;
		myfile_OC << endl;
	}

	ofstream myfile_out2;
	myfile_out2.precision(10);
	if(out2==1)
		myfile_out2.open("(m_F_numberofcells)foreverycohort.m", ios::app);

	ofstream myfile;
	myfile.precision(10);
	myfile.open("data_figd3_fp.txt", ios::app);

	int p = 1; //number of the cohort in list coh
	for(cohit=coh.begin();cohit!=coh.end();++cohit){ //integrates the given list of cohorts coh
		if(out==1)
			myfile_OC << "cohort " << p << " with initial mass " << cohit->getMass() << endl;
		if(out2==1){
			myfile_out2 << "coh" << p << " = [";
			myfile_out2.close();
		}
		i_sm = 0;
		i_gr = 0;
		how_sm = 0;
		how_gr = 0;
		if(cohit->getb()>0){ //integrate the cohort IF it is non-empty
			cohit->integrate_cohort_dest(out,out2,i_sm,how_sm,i_gr,how_gr); 
			con += cohit->getb()*cohit->getTheta(); //calculate the amount of nutrient consumed by the cells in the cohort and adds it to con 
		}
		else{
			if(out==1)
				myfile_OC << "no cells in this cohort" << endl;
		}
		if(out2==1){
			myfile_out2.open("(m_F_numberofcells)foreverycohort.m", ios::app);
			myfile_out2 << "]" << endl;
		}
		
		/*for the small daughter cells*/
		if(how_sm==0) //the initial cohort is empty or there is a mistake in the code, this should not happen normally
			destination_sm(p-1) = 0;
		else if(how_sm==1){
			destination_sm(p-1) = i_sm+1;
			/*all integers already in destination_sm and destination_gr (so at a location smaller than p-1) 
			that are larger then i_sm, should be increased by 1*/
			for(int i=0; i<p-1; i++){
				if(destination_sm(i)>i_sm)
					destination_sm(i) = destination_sm(i)+1;
				if(destination_gr(i)>i_sm)
					destination_gr(i) = destination_gr(i)+1;
			}
		}
		else if(how_sm==2){
			destination_sm(p-1) = i_sm;
			/*no changes in integers already in destination_sm and destination_gr needed*/
		}
		else if(how_sm==3){
			destination_sm(p-1) = i_sm;
			/*all integers already in destination_sm and destination_gr (so at a location smaller than p-1) 
			that are larger then i_sm, should be decreased by 1*/
			for(int i=0; i<p-1; i++){
				if(destination_sm(i)>i_sm)
					destination_sm(i) = destination_sm(i)-1;
				if(destination_gr(i)>i_sm)
					destination_gr(i) = destination_gr(i)-1;
			}
		}
		else if(how_sm==-1){ // the cells in the cohort don't divide
			destination_sm(p-1) = -1;
		}
		else //this should not happen normally, there is a mistake in the code
			destination_sm(p-1) = -2;


		/*for the big daughter cells*/
		if(how_gr==0) //the initial cohort is empty or there is a mistake in the code, this should not happen normally
			destination_gr(p-1) = 0;
		else if(how_gr==1){
			destination_gr(p-1) = i_gr+1;
			/*all integers already in destination_sm and destination_gr (so at a location smaller than p-1) 
			that are larger then i_gr, should be increased by 1*/
			for(int i=0; i<p-1; i++){
				if(destination_sm(i)>i_gr)
					destination_sm(i) = destination_sm(i)+1;
				if(destination_gr(i)>i_gr)
					destination_gr(i) = destination_gr(i)+1;
			}
		}
		else if(how_gr==2){
			destination_gr(p-1) = i_gr;
			/*no changes in integers already in destination_sm and destination_gr needed*/
		}
		else if(how_gr==3){
			destination_gr(p-1) = i_gr;
			/*all integers already in destination_sm and destination_gr (so at a location smaller than p-1) 
			that are larger then i_gr, should be decreased by 1*/
			for(int i=0; i<p-1; i++){
				if(destination_sm(i)>i_gr)
					destination_sm(i) = destination_sm(i)-1;
				if(destination_gr(i)>i_gr)
					destination_gr(i) = destination_gr(i)-1;
			}
		}
		else if(how_gr==-1){ // the cells in the cohort don't divide
			destination_gr(p-1) = -1;
		}
		else //this should not happen normally, there is a mistake in the code
			destination_gr(p-1) = -2;

		p++;
	}

	if(out2==1)
		myfile_out2.close();

	realtype eps = RCONST(eps_b*0.01*totaalb); //calculate eps = the minimum amount of cells in a cohort so that the cohort can't be neglected
	                                           // = eps_b% of total amount of cells born
	if(out==1){
		myfile_OC << "total amount of cells born = " << totaalb << endl;
		myfile_OC << "epsilon = " << eps << endl;
	}

	/*loop over the list of created cohorts (cohnew): remove cohorts with b<eps and calculate the total amount of cells in the remaining cohorts (totalb)*/
	p = 1;
	if(out==1)
		myfile_OC << "overview of new cohorts" << endl;
	cohit = cohnew.begin();
	while(cohit!=cohnew.end()){
		if(cohit->getb()>=eps){
			totalb += cohit->getb();
			if(out==1)
				myfile_OC << "cohort " << p << " with initial mass " << cohit->getMass() << " CycB_T " << cohit->getCycB_T() << " Cdh1 " << cohit->getCdh1() << " Cdc20_T " << cohit->getCdc20_T() << " Cdc20_A " << cohit->getCdc20_A() << " IEP " << cohit->getIEP() << " CKI_T " << cohit->getCKI_T() << " SK " << cohit->getSK() << " TF " << cohit->getTF() << " number " << cohit->getb() <<endl;
			p++;
			cohit++;
		}
		else{
			if(out==1)
				myfile_OC << "cohort erased with initial mass " << cohit->getMass() << " because amount of cells smaller than epsilon (" << cohit->getb() << ")" << endl;
			cohit = cohnew.erase(cohit);
		}
	}

	tempS = RCONST(S+prop*(RCONST(S0-(1.0/D)*con)-S)); //calculate new S for the next age integration

	if(out==1){
		myfile_OC << "total births in all cohorts: " << totalb << endl;
		myfile_OC << "total amount of nutrient used by all the cohorts: " << con << endl;
		myfile_OC << "intrinsic change of nutrient per unit of time " << D*(S0-S) << endl;
		myfile_OC << "nutrient concentration for next age integration (unadjusted) " << RCONST(S0-(1.0/D)*con) << endl;
		myfile_OC << "nutrient concentration for next age integration (adjusted) " << RCONST(S+prop*(RCONST(S0-(1.0/D)*con)-S)) << endl;
		if(RCONST(S+prop*(RCONST(S0-(1.0/D)*con)-S))<0.000001)
			myfile_OC << "nutrient concentration for next age integration (adjusted) too small, so set S equal to 0.000001" << endl;
		myfile_OC << endl;
	}

	//print the output-file data_figd3_fp.txt

	//save vector with the masses of the cohorts
	alglib::real_1d_array destination_mass; //vector with the masses of the daughter cells
	destination_mass.setlength(coh.size());
	int i = 0;
	for(cohit=coh.begin();cohit!=coh.end();++cohit){ 
		destination_mass(i) = cohit->getMass();
		i++;
	}
	
	cohit = coh.begin();
	myfile << "massa,massa_kl,massa_gr,gewicht" << endl;
	for(i=0;i<coh.size();i++){
		myfile << cohit->getMass() << "," << destination_mass(destination_sm(i)-1) << "," << destination_mass(destination_gr(i)-1) << "," << cohit->getb() << endl;
		cohit++;
	}
	myfile.close();

}

void loopMap(realtype Sp, clist &coh, int k, int out, int file_numberofcells){ 
	/* k repeated evaluations of the map with adjustement of S and coh after each iteration (for fixed S0=S0p)
	   results: adjusted list of cohorts in coh and adjusted Sp in S (S globally defined)
	   parameter out (int): =1 (full output), = 0 (only important output)
	   parameter file_numberofcells (int): = 1 (create a matlab-file numberofcells.m for the last iteration of the map with
	                                            for every cohort a matrix with the columns m; surv. prob.; number of cells alive) 
										   = 0 (creates no such file) */

	S = Sp; //set S to given value Sp
	int out2 = 0; //out2 is used to determine if matlab-file numberofcells.m has to be created 
	              //(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
	              //if == 0: no matlab-file is created, if ==1: the file is created
				  //here it is only used for the last loop of the map

	clistit cohit;
	ofstream myfile;
	myfile.precision(10);
	
	if(out==1)
		myfile_OC << "S = " << S << endl << endl;

	/*loop of k evaluations of the map with adjustement of S and coh after each iteration*/
	for(int j=1;j<=k;j++){ 
		if(out==1)
			myfile_OC << "age integration " << j << endl << endl;
		if(file_numberofcells==1){
			if(j==k)
				out2 = 1; //creates file numberofcells.m for last loop of the map
		}
		integrate_allcohorts(coh,out,out2); //evaluation of the map, results in new birth cohorts in list cohnew, new S in tempS
		
		/*print output in file data.txt: per iteration one line: the number of the iteration, the number of cohorts created (so in cohnew), 
		                                                the new value of S, for every cohort m CycB_T ... TF b, total amount of cells created*/
		myfile.open ("data.txt", ios::app);
		myfile << j << ", " << cohnew.size() << "," << tempS << ",";
		for(cohit=cohnew.begin();cohit!=cohnew.end();++cohit)
			myfile << cohit->getMass() << "," << cohit->getCycB_T() << "," << cohit->getCdh1() << "," << cohit->getCdc20_T() << "," << cohit->getCdc20_A() << "," << cohit->getIEP() << "," << cohit->getCKI_T() << "," << cohit->getSK() << "," << cohit->getTF() << "," << cohit->getb() << ",";
		myfile << totalb << endl;
		myfile.close();

		/*print output in file NSb.txt: all output on the same line: the number of the iteration,
		                                   the number of cohorts created (in cohnew), the new value of S, total amount of cells created,*/
		myfile.open("NSb.txt",ios::app);
		myfile << j << " " << cohnew.size() << " " << tempS << " " << totalb << endl;
		myfile.close();

		/*if the new value of S is negative or close to zero (<0.000001): set S to 0.000001, otherwise it is unchanged (so equal to tempS)*/
		if(tempS<0.000001)
			S = RCONST(0.000001);
		else
			S = tempS;

		coh.assign(cohnew.begin(),cohnew.end()); //put the new list of cohorts in coh
	}

	/*empty the list of cohorts cohnew*/
	cohit=cohnew.begin();
	while(cohit!=cohnew.end())
		cohit=cohnew.erase(cohit);
}

void loopMap_C(realtype Sp, clist &coh, int &k, realtype reltol_S, realtype reltol_btot, int out, int file_numberofcells){
	/* repeated evaluations of the map with adjustement of S and coh after each iteration
	   -> evaluations repeated until convergence (when the relative adjustment to S is smaller than reltol_S
	                   and the relative adjustment to btot is smaller than reltol_btot and the number of cohorts is constant)
		-> maximally k map iterations 
	   results: adjusted list of cohorts in coh and adjusted Sp in S (S globally defined) and effective number of evaluations in k
	   parameter out (int): =1 (full output), = 0 (only important output) 
	   parameter file_numberofcells (int): = 1 (create a matlab-file numberofcells.m for the last iteration of the map with
	                                            for every cohort a matrix with the columns m; surv. prob.; number of cells alive) 
										   = 0 (creates no such file) */

	S = Sp; //set S to given value Sp
	int out2 = 0; //out2 is used to determine if matlab-file numberofcells.m has to be created 
	              //(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
	              //if == 0: no matlab-file is created, if ==1: the file is created
				  //here it is only used for the last loop of the map
	clistit cohit;
	ofstream myfile;
	myfile.precision(10);

	//to check convergence of the map, we have to compare N, btot and S of 2 consecutive map iterations
	int N_1 = coh.size();
	int N_2;
	realtype btot_1 = RCONST(0);
	for(cohit=coh.begin();cohit!=coh.end();++cohit)
		btot_1 += cohit->getb();
	realtype btot_2;
	realtype S_1 = Sp;
	realtype S_2;
	
	if(out==1)
		myfile_OC << "S = " << S << endl << endl;

	// do 1 map iteration
	if(out==1)
			myfile_OC << "age integration " << 1 << endl << endl;
	if(file_numberofcells==1){
		if(k==1)
			out2 = 1;
	}
	integrate_allcohorts(coh,out,out2); //gives new birth cohorts in list cohnew, new S in tempS
	/*print output in file data.txt: per iteration one line: the number of the iteration, the number of cohorts created (so in cohnew), 
		                                            the new value of S, for every cohort m CycB_T ... TF b, total amount of cells created*/
	myfile.open ("data.txt", ios::app);
	myfile << 1 << ", " << cohnew.size() << "," << tempS << ",";
	for(cohit=cohnew.begin();cohit!=cohnew.end();++cohit)
		myfile << cohit->getMass() << "," << cohit->getCycB_T() << "," << cohit->getCdh1() << "," << cohit->getCdc20_T() << "," << cohit->getCdc20_A() << "," << cohit->getIEP() << "," << cohit->getCKI_T() << "," << cohit->getSK() << "," << cohit->getTF() << "," << cohit->getb() << ",";
	myfile << totalb << endl;
	myfile.close();
	/*print output in file NSb.txt: all output on the same line: the number of the iteration,
		                                the number of cohorts created (in cohnew), the new value of S, total amount of cells created,*/
	myfile.open("NSb.txt",ios::app);
	myfile << 1 << " " << cohnew.size() << " " << tempS << " " << totalb << endl;
	myfile.close();
	/*if the new value of S is negative or close to zero (<0.000001): set S to 0.000001, otherwise it is unchanged (so equal to tempS)*/
	if(tempS<0.000001)
		S = RCONST(0.000001);
	else
		S = tempS;
	coh.assign(cohnew.begin(),cohnew.end()); //put the new list of cohorts in coh
	N_2 = coh.size();
	btot_2 = totalb;
	S_2 = S;

	int diff_N = floor(N_1*var_N); //maximum allowed change in number of cohorts from one map iteration to the next in order to have convergence of the map

	int i = 1; //number of map iterations completed
	/*loop of evaluations of the map with adjustement of S and coh after each iteration
	  as long as conditions for convergence are not fulfilled AND number of iterations done < k
	  (conditions for convergence: the difference in number of cohorts, compared to the previous result of the map iteration, is maximally diff_N,
	                         AND S changes less than reltol_S compared to the previous result of the map iteration,
							 AND btot changes less than reltol_btot compared to the previous map iteration) */
	while(((abs(N_2-N_1)>diff_N) || ((abs(btot_1-btot_2)/btot_1)>=reltol_btot) || ((abs(S_1-S_2)/S_1)>=reltol_S)) 
		        && (i<k)){
		if(out==1)
			myfile_OC << "age integration " << i+1 << endl << endl;
		if(file_numberofcells==1){
			if((i+1)==k)
				out2 = 1;
		}
		integrate_allcohorts(coh,out,out2); //gives new birth cohorts in list cohnew, new S in tempS
		/*print output in file data.txt: per iteration one line: the number of the iteration, the number of cohorts created (so in cohnew), 
		                                                the new value of S, for every cohort m CycB_T ... TF b, total amount of cells created*/
		myfile.open ("data.txt", ios::app);
		myfile << i+1 << ", " << cohnew.size() << "," << tempS << ",";
		for(cohit=cohnew.begin();cohit!=cohnew.end();++cohit)
			myfile << cohit->getMass() << "," << cohit->getCycB_T() << "," << cohit->getCdh1() << "," << cohit->getCdc20_T() << "," << cohit->getCdc20_A() << "," << cohit->getIEP() << "," << cohit->getCKI_T() << "," << cohit->getSK() << "," << cohit->getTF() << "," << cohit->getb() << ",";
		myfile << totalb << endl;
		myfile.close();

		/*print output in file NSb.txt: all output on the same line: the number of the iteration,
		                                   the number of cohorts created (in cohnew), the new value of S, total amount of cells created,*/
		myfile.open("NSb.txt",ios::app);
		myfile << i+1 << " " << cohnew.size() << " " << tempS << " " << totalb << endl;
		myfile.close();

		/*if the new value of S is negative or close to zero (<0.000001): set S to 0.000001, otherwise it is unchanged (so equal to tempS)*/
		if(tempS<0.000001)
			S = RCONST(0.000001);
		else
			S = tempS;

		coh.assign(cohnew.begin(),cohnew.end()); //put the new list of cohorts in coh

		N_1 = N_2;
		btot_1 = btot_2;
		S_1 = S_2;
		N_2 = coh.size();
		btot_2 = totalb;
		S_2 = S;
		i++;
		diff_N = floor(N_1*var_N);

		if(out==1){
			myfile_OC << "number of cohorts in previous step = " << N_1 << ", number of cohorts now = " << N_2 << endl;
			myfile_OC << "=> change in number of cohorts = " << abs(N_2-N_1) << ", compared to maximally allowed change = " << diff_N << endl;
			myfile_OC << "S in previous step = " << S_1 << ", S now = " << S_2 << endl;
			myfile_OC << "relative change in S (relative to previous step) = " << (abs(S_1-S_2)/S_1) << " compared to reltol_S = " << reltol_S << endl;
			myfile_OC << "b_tot in previous step = " << btot_1 << ", b_tot now = " << btot_2 << endl;
			myfile_OC << "relative change in btot (relative to previous step) = " << (abs(btot_1-btot_2)/btot_1) << " compared to reltol_btot = " << reltol_btot << endl;
			myfile_OC << endl;
		}
	}

	k = i; // the effective number of iterations of the map done

	/*empty the list of cohorts cohnew*/
	cohit=cohnew.begin();
	while(cohit!=cohnew.end())
		cohit=cohnew.erase(cohit);
}

int main() {
	myfile_OC.precision(10);
	myfile_OC.open ("outputcohorts.txt", ios::app);

	clist coh;
	cohort a;
	a.set_init(0.301425,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.01);
	coh.push_back(a);
	int k = 500;
	loopMap_C(1.0,coh,k,1.0e-7,1.0e-7,1,0);

	myfile_OC.close();
	return 0;
}

realtype nu(N_Vector y){ //rate of individual mortality
	return RCONST(D);
}

realtype beta(N_Vector y){
	/*rate of reproduction
	now set to 0 since all the cells in the cohorts divide at the same age (when certain conditions are fulfilled)
	this can be used later for a more general beta-function*/
	realtype bet = RCONST(0);
	return bet;
}

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data){ /*ODEs for m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,F,theta*/
	realtype m, CycB_T, Cdh1, Cdc20_T, Cdc20_A, IEP, CKI_T, SK, TF, F, theta;
	m = NV_Ith_S(y,0);
	CycB_T = NV_Ith_S(y,1);
	Cdh1 = NV_Ith_S(y,2);
	Cdc20_T = NV_Ith_S(y,3);
	Cdc20_A = NV_Ith_S(y,4);
	IEP = NV_Ith_S(y,5);
	CKI_T = NV_Ith_S(y,6);
	SK = NV_Ith_S(y,7);
	TF = NV_Ith_S(y,8);
	F = NV_Ith_S(y,NDIM);
	theta = NV_Ith_S(y,NDIM+1);
	realtype sigma = RCONST(1.0/Keq) + CycB_T + CKI_T;
	realtype Trimer = RCONST((2.0*CycB_T*CKI_T)/(sigma + sqrt(pow(sigma,2)-4.0*CycB_T*CKI_T)));
	realtype CycB = RCONST(CycB_T-Trimer);
	NV_Ith_S(ydot,0) = RCONST(mu)*m*(1-m/RCONST(m_max))*RCONST(c1)*S/(RCONST(zeta1)+S);
	NV_Ith_S(ydot,1) = RCONST(k1) - (RCONST(k2p)+RCONST(k2pp)*Cdh1+RCONST(k2ppp)*Cdc20_A)*CycB_T;
	NV_Ith_S(ydot,2) = (RCONST(k3p)+RCONST(k3pp)*Cdc20_A)*(1.0-Cdh1)/(RCONST(J3)+1.0-Cdh1) - ((RCONST(k4)*m*CycB+RCONST(k4p)*SK)*Cdh1/(RCONST(J4)+Cdh1));
	NV_Ith_S(ydot,3) = RCONST(k5p)+(RCONST(k5pp)*pow(m*CycB,n))/(pow(RCONST(J5),n)+pow(m*CycB,n))-RCONST(k6)*Cdc20_T;
	NV_Ith_S(ydot,4) = ((RCONST(k7)*IEP*(Cdc20_T-Cdc20_A))/(J7+Cdc20_T-Cdc20_A)) - (RCONST(k8*Mad*Cdc20_A)/(RCONST(J8)+Cdc20_A)) - RCONST(k6)*Cdc20_A;
	NV_Ith_S(ydot,5) = (RCONST(k9)*m*CycB*(1.0-IEP)) - (RCONST(k10)*IEP);
	NV_Ith_S(ydot,6) = RCONST(k11) - ((RCONST(k12p)+RCONST(k12pp)*SK+RCONST(k12ppp)*m*CycB)*CKI_T);
	NV_Ith_S(ydot,7) = RCONST(k13p) + RCONST(k13pp)*TF - RCONST(k14)*SK;
	NV_Ith_S(ydot,8) = ((RCONST(k15p)*m+RCONST(k15pp)*SK)*(1.0-TF))/(RCONST(J15)+1.0-TF) - ((RCONST(k16p)+RCONST(k16pp)*m*CycB)*TF)/(RCONST(J16)+TF);
	NV_Ith_S(ydot,NDIM) = -(nu(y)+beta(y))*F;
	NV_Ith_S(ydot,NDIM+1) = RCONST(c2)*RCONST(mu)*m*(1-m/RCONST(m_max))*RCONST(c1)*S/(RCONST(zeta1)+S);
	return(0);
}

static int g(realtype t, N_Vector y, realtype *gout, void *user_data){ //for RootFinding during CVODE integration
  gout[0] = NV_Ith_S(y,1)-X_div;
  return(0);
}

void cohort::PrintOutput(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6, realtype y7, realtype y8, realtype y9, realtype y10, realtype y11){
	/*prints the data values (t,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,F,theta) of a cohort at a certain age t in the file outputcohorts.txt*/
  myfile_OC << "At t = " << t << " m = " << y1 << " CycB_T = " << y2 << " Cdh1 = " << y3 << " Cdc20_T = " << y4 << " Cdc20_A = " << y5 << " IEP = " << y6 << " CKI_T = " << y7 << " SK = " << y8 << " TF = " << y9 <<" F = " << y10 << " theta = " << y11 << endl;
  return;
}

void cohort::PrintOutputData(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6, realtype y7, realtype y8, realtype y9, realtype y10, realtype y11){
	/*prints the data values (t,m,CycB_T,Cdh1,Cdc20_T,Cdc20_A,IEP,CKI_T,SK,TF,F,theta) of a cohort at a certain age t in the file values.txt*/
  ofstream myfile;
  myfile.open ("values.txt", ios::app);
  myfile.precision(10);
  myfile << t << " " << y1 << " " << y2 << " " << y3 << " " << y4 << " " << y5 << " " << y6 << " " << y7 << " " << y8 << " " << y9 << " " << y10 << " " << y11 << endl;
  myfile.close();
  return;
}

void printListCoh_OC(clist &coh){ /*prints the given list of cohorts in outputcohorts.txt*/
	clistit cohit;
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		myfile_OC << cohit->getMass() << " " << cohit->getCycB_T() << " " << cohit->getCdh1() << " " << cohit->getCdc20_T() << " " << cohit->getCdc20_A() << " " << cohit->getIEP() << " " << cohit->getCKI_T() << " " << cohit->getSK() << " " << cohit->getTF() << " " << cohit->getb() << " " << endl;
	}
}

int cohort::check_flag(void *flagvalue, char *funcname, int opt){ /*handles CVODE errors*/
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
    return(1);
  }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",funcname, *errflag);
	  return(1); 
	}
  }

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",funcname);
    return(1);
  }

  return(0);
}