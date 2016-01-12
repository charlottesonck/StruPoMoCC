/*code accompanying the PhD "Structured population models with internal cell cycle" by Charlotte Sonck
  Toy model of Tyson and Novák used as internal structure for the cells
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
#define cond_div 0 /*0 if X_div is constant (typically 0.1), 1 if discrete values for X_div dependent on m_birth*/
#define cond_div2 0 /*0 if X_div is constant (typically 0.1), 1 if probability distribution for X_div*/

#define NDIM 4  /*dimension chemical submodel for the cell cycle*/
#define phi 0.4  /*parameter for mass division*/
#define m_min 0.75 /*minimum mass for division*/
#define zeta1 0.5 /*parameter for dependency mass increase on nutrient*/
#define c1 2 /*parameter for dependency mass increase on nutrient*/
#define c2 1 /*parameter consumption rate*/
realtype S0=RCONST(1); /*concentration nutrient in feeding bottle*/
realtype D=RCONST(0.01); /*diffusion rate*/
#define Ft 1.0e-7 /*lower threshold for survival probability F*/

/*cell cycle model parameters*/
#define k1 0.04
#define k2p 0.04
#define k2pp 1
#define k3p 1
#define k3pp 10
#define k4 35
#define J3 0.04
#define J4 0.04
#define k5p 0.005
#define k5pp 0.2
#define k6 0.1
#define J5 0.3
#define n 4
#define mu 0.01
#define m_max 10
realtype X_div=RCONST(0.1); /*division threshold for X*/

/*definitions related to the age integration*/
#define T0 RCONST(0.0) /*initial age*/
#define T1 RCONST(0.5) /*first output age*/
#define RELTOL RCONST(1.0e-6) /*scalar relative tolerance*/
#define ABSTOL RCONST(1.0e-8) /*scalar absolute tolerance*/
#define NEQ 6 /*number of equations (m,X,Y,A,F,theta)*/
realtype TIN=RCONST(0.5); /*size agestep*/
#define eps_m RCONST(1.0e-7) /*relative to m_max, how close in the integration we maximally go to m_max*/
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data); /*functions for age integration*/
static int g(realtype t, N_Vector y, realtype *gout, void *user_data); /*functions for RootFindingFunction during age integration*/

/*definitions related to the class cohort*/
class cohort;
typedef list <cohort> clist;
typedef list <cohort>:: iterator clistit;
#define delta RCONST(1.0e-3) /*"inhibition zone" cohort*/
#define eps_b RCONST(1.0e-10) /*eps_b = epsilon = value (in %, so 1 means 1%) of total number of cells born that can be neglected*/
#define var_N RCONST(0.01) /*used in function loopMap_C: relative amount of variation allowed in the number of cohorts at convergence
                           for example 0.01 means that N has to change less than 0.01*N from one iteration to the next, 
						   in order to say that the map has converged to a fixed point*/

void integrate_allcohorts(clist coh, int out, int out2); /*integrates a given list of cohorts coh*/
void integrate_allcohorts_dest(clist coh, int out, int out2); /*integrates a given list of cohorts coh and prints data_figd3_fp.txt
															  with the information needed for the cohort-to-cohort representation*/
void addcohort(realtype p, clist &cohnew, realtype in, realtype m, realtype X, realtype Y, realtype A, int out); 
                        /*adds the daughter cells of a dividing cohort (with m, X, Y,A  and in cells) to a list of cohorts cohnew*/
void printListCoh_OC(clist &coh); /*prints the given list of cohorts in outputcohorts.txt*/

/*global variables*/
clist cohnew; /*list of new cohorts created by the integration of a list of cohorts (integrate_allcohorts)*/
clist cohadj; /*in function mapM_B: list of cohorts created by the mapping of a list of cohorts (coh_new) to the original list of cohorts (coh_or)*/
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
int how_gr = 0; /*idem for the large daughter cells*/

class cohort{
	realtype init[NDIM]; /*birth state vector*/
	realtype F; /*survival chance of cells in the cohort*/
	realtype theta; /*amount of nutrient consumed by one cell in the cohort up to its current age*/
	realtype b; /*original number of cells in the cohort*/
public:
	void set_init (realtype mass, realtype X, realtype Y, realtype A, realtype B); /*initializes the state vector of a cohort: mass, X, Y, A and number of cells*/
	void setb(realtype B) {b=B;}; /*initializes b (the original number of cells in the cohort)*/
	void setm(realtype m) {init[0]=m;}; /*changes the mass of the cohort*/
	void setTheta (realtype t) {theta=t;} /*changes the value of theta*/
	realtype getb() {return b;} /*returns the original number of cells*/
	realtype getTheta() {return theta;} /*returns theta*/
	realtype getMass() {return init[0];} /*returns the mass*/
	realtype getX() {return init[1];} /*returns X*/
	realtype getY() {return init[2];} /*returns Y*/
	realtype getA() {return init[3];} /*returns A*/
	realtype getF() {return F;} /*returns the survival probability*/
	int integrate_cohort (int out, int out2);  /*integrates the cohort until the age where F < Ft*/
	int integrate_cohort_bis (int out, int out2);  /*integrates the cohort until the age where F < Ft for cond_div2==1 (probability distribution for X_div)*/
	int integrate_cohort_dest (int out, int out2, int &i_sm, int &how_sm, int &i_gr, int &how_gr); /*integrates the cohort until the age where F < Ft 
																								   + gives information needed for the cohort-to-cohort-representation*/
	void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6); 
	           /*prints the data values (t,m,X,Y,A,F,theta) of a cohort at a certain age t in the file outputcohorts.txt*/
	void PrintOutputData(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6);
	           /*prints the data values (t,m,X,Y,A,F,theta) of a cohort at a certain age t in the file values.txt*/
	int check_flag(void *flagvalue, char *funcname, int opt); /*handles CVODE errors*/
};

void cohort::set_init(realtype mass, realtype X, realtype Y, realtype A, realtype B) {
	/*initializes the state vector of a cohort: mass, X, Y, A and number of cells*/
	init[0] = mass;
	init[1] = X;
	init[2] = Y;
	init[3] = A;
	F = RCONST(1.0); //set the survival probability to 1 
	theta = RCONST(0.0); //set the amount of nutrient consumed by one cell in the cohort up to its current age equal to 0
	b = B; //set the original number of cells in the cohort equal to B
}

int cohort::integrate_cohort(int out, int out2) { 
	/* integrates the cohort until the age where F < Ft
	   if cohorts are created by division, they are added to the list cohnew (globally defined list of cohorts)
	   after each age integration step, we check if cells divided during this step (due to the function beta)
	          and whether the conditions are fulfilled that say that all the remaining cells in the cohort should divide
	          -> when the latter conditions are fulfilled, all remaining cells in the cohort divide
	   parameter out (int): = 1 (print full output), = 0 (print only the most important output) 
	   parameter out2 (int): used to determine if Matlabfile numberofcells.m has to be created 
			(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
			= 0: no matlab-file is created, = 1: the file is created*/

	realtype m, X, Y, A; /*the values of m,X,Y,A of the cells in the cohort at the current age integration output step*/
	realtype F_old, m_old, X_old, Y_old, A_old; /*the values of F,m,X,Y,A of the cells in the cohort at the previous age integration output step*/
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
		PrintOutput(iout,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
	realtype toutprev = 0; //save the previous output time in toutprev

	if(cond_div==1){ //if discrete values for X_div dependent on m_birth are used, determines the value of X_div for the cohort
		int j = 1;
		realtype mesh = RCONST(0.3+j*0.01);
		while((NV_Ith_S(y,0)>mesh) && (j<71)){
			j++;
			mesh = RCONST(0.3+j*0.01);
		}
		if(j!=71)
			X_div = RCONST(0.08+(j-1)*0.00055);
		else //so j == 71
			X_div = 0.11905;
		if(out==1){
			myfile_OC << "m_birth = " << NV_Ith_S(y,0) << " is not bigger than " << mesh << endl;
			myfile_OC << "so X_div is equal to " << X_div << " (j= " << j << ")"<< endl;
		}
	}

	/*booleans vwm_min, vwm_max0 and vwm_max1 are defined to handle the different cases 
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
			&& ((NV_Ith_S(y,1))<=(X_div+1.0e-6))              // X in [X_div-1.0e-6,X_div+1.0e-6]
			&& k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1)<0   //  and dX/dt<0
			&& vwm_min                                       //  and m>=m_min if cond_min is 1 (always true otherwise)
			&& vwm_max0                                      //  and m<=(m_max*(1-eps_m)) if cond_max is 0 (always true otherwise)
			&& vwm_max1){                                    //  and m<=m_max if cond_max is 1 (always true otherwise)
		if(out==1){
			myfile_OC << "Conditions for division are immediately fulfilled: all cells divide immediately" << endl;
			myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
		}
		m = NV_Ith_S(y,0);
		X = NV_Ith_S(y,1);
		Y = NV_Ith_S(y,2);
		A = NV_Ith_S(y,3); 
		theta = NV_Ith_S(y,NDIM+1); /*amount of nutrient consumed by one cell in the cohort up to its current age (when it divides)*/
		realtype in = NV_Ith_S(y,NDIM)*b; /*the number of cells that divide*/
		addcohort(phi,cohnew,in,m,X,Y,A,out); /*let the cells divide: add the resulting cohorts to the list cohnew*/
		if(out==1){
			myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
			printListCoh_OC(cohnew);
		}
	}
	else{ // when cells in the cohort don't all divide immediately
		while((Fto0!=1) //repeat age integration step on the cohort while: conditions for division of the remaining cells are not fulfilled                 
				&& vwm_max0                    // and  m<=(m_max*(1-eps_m)) if cond_max is 0 (always true otherwise)
				&& vwm_max1                    // and  m<=m_max if cond_max=1 (always true otherwise)
				&& (NV_Ith_S(y,NDIM)>=Ft)){    // and  survival probability of the cells in the cohort >= Ft
			/*save the values of F,m,X,Y,A of the cells in the cohort at the previous age integration output step*/
			F_old = NV_Ith_S(y,NDIM);
			m_old = NV_Ith_S(y,0);
			X_old = NV_Ith_S(y,1);
			Y_old = NV_Ith_S(y,2);
			A_old = NV_Ith_S(y,3);
			
			if(toutprev==0){
				if((NV_Ith_S(y,1)>=(X_div-1.0e-6)) && (NV_Ith_S(y,1)<=(X_div+1.0e-6))){ 
					// if age integration starts in the root X==X_div: take age integration step without RootFinding
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
						PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
					if(out2==1)
						myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
					Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
																//ATTENTION: only valid when individual mortality function nu = D
					toutprev = t; //save current output time in toutprev 
					tout += TIN; //next output time

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
						//let these cells (with state values m_old,X_old,Y_old,A_old) divide: add the resulting cohorts to the list cohnew
						addcohort(phi,cohnew,in,m_old,X_old,Y_old,A_old,out); 
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}

					//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
					if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                        // if X in [X_div-1.0e-6,X_div+1.0e-6]
							&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
							&& (k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1) < 0)    // and dX/dt<0
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
						X = NV_Ith_S(y,1);
						Y = NV_Ith_S(y,2);
						A = NV_Ith_S(y,3);
						theta = NV_Ith_S(y,NDIM+1);
						realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
						addcohort(phi,cohnew,in,m,X,Y,A,out);
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}
					F_old = NV_Ith_S(y,NDIM);
					m_old = NV_Ith_S(y,0);
					X_old = NV_Ith_S(y,1);
					Y_old = NV_Ith_S(y,2);
					A_old = NV_Ith_S(y,3);

					CVodeFree(&cvode_mem);
				}

				if((Fto0!=1) //do age integration step on the cohort if conditions for division of the remaining cells are not fulfilled                 
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
					PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
				if(out2==1)
					myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
				Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
															//ATTENTION: only valid when individual mortality function nu = D
				toutprev = t; //save current output time in toutprev 
				tout+=TIN; //next output time

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
					//let these cells (with state values m_old,X_old,Y_old,A_old) divide: add the resulting cohorts to the list cohnew
					addcohort(phi,cohnew,in,m_old,X_old,Y_old,A_old,out); 
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}

				//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
				if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                      // if X in [X_div-1.0e-6,X_div+1.0e-6]
						&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
						&& (k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1) < 0)  // and dX/dt<0
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
					X = NV_Ith_S(y,1);
					Y = NV_Ith_S(y,2);
					A = NV_Ith_S(y,3);
					theta = NV_Ith_S(y,NDIM+1);
					realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
					addcohort(phi,cohnew,in,m,X,Y,A,out);
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
	/*merges 2 cohorts in the list of cohorts cohnew: the cohort at cohnewit and the one at cohnewit+1
	  weighted means are used for the values of m, X, Y and A
	  resulting list of cohorts in cohnew and the pointer cohnewit points to the resulting merged cohort
	  parameter out (int): = 1 (give full output), = 0 (give only important output) */

	if(out==1)
		myfile_OC << "merge two cohorts together" << endl;

	/*store the values of the left cohort in left_m, left_X, left_Y, left_A, left_b*/
	realtype left_m = cohnewit->getMass();
	realtype left_X = cohnewit->getX();
	realtype left_Y = cohnewit->getY();
	realtype left_A = cohnewit->getA();
	realtype left_b = cohnewit->getb();
	if(out==1)
		myfile_OC << "merge left cohort: " << left_m << " " << left_X << " " << left_Y << " " << left_A << " " << left_b << endl;

	cohnewit = cohnew.erase(cohnewit); //erase the left cohort, right cohort now at position cohnewit

	/*merge the cells of the left cohort with the cells in the right cohort at cohnewit*/
	realtype old_m = cohnewit->getMass();
	realtype old_X = cohnewit->getX();
	realtype old_Y = cohnewit->getY();
	realtype old_A = cohnewit->getA();
	realtype old_b = cohnewit->getb();
	if(out==1)
		myfile_OC << "with right cohort: " << old_m << " " << old_X << " " << old_Y << " " << old_A << " " << old_b << endl;
	realtype new_m = RCONST((old_b*old_m+left_b*left_m)/(old_b+left_b));
	realtype new_X = RCONST((old_b*old_X+left_b*left_X)/(old_b+left_b));
	realtype new_Y = RCONST((old_b*old_Y+left_b*left_Y)/(old_b+left_b));
	realtype new_A = RCONST((old_b*old_A+left_b*left_A)/(old_b+left_b));
	realtype new_b = RCONST(old_b+left_b);
	if(out==1)
		myfile_OC << "resulting cohort: " << new_m << " " << new_X << " " << new_Y << " " << new_A << " " << new_b << endl;

	cohnewit->set_init(new_m,new_X,new_Y,new_A,new_b); //change the values of the right cohort to these new "merged values"
}

void merge_Coh(clist &cohnew, clistit &cohnewit, realtype cell_m, realtype cell_X, realtype cell_Y, realtype cell_A, realtype cell_b, int out){
	/*merge cell_b cells (with mass cell_m, X cell_X, Y cell_Y and A cell_A) with the cohort at location cohnewit in the list cohnew
	  weighted means are used for the values of m, X, Y and A
	  resulting list of cohorts in cohnew and the pointer cohnewit points to the resulting merged cohort
	  parameter out (int): = 1 (give full output), = 0 (give only important output) */

	if(out==1)
		myfile_OC << "merge the cells: " << cell_m << " " << cell_X << " " << cell_Y << " " << cell_A << " " << cell_b << endl;
	/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit (old_m,old_X,old_Y,old_A,old_b)*/
	realtype old_m = cohnewit->getMass();
	realtype old_X = cohnewit->getX();
	realtype old_Y = cohnewit->getY();
	realtype old_A = cohnewit->getA();
	realtype old_b = cohnewit->getb();
	if(out==1)
		myfile_OC << "with the cohort: " << old_m << " " << old_X << " " << old_Y << " " << old_A << " " << old_b << endl;
	realtype new_m = RCONST((old_b*old_m+cell_b*cell_m)/(old_b+cell_b));
	realtype new_X = RCONST((old_b*old_X+cell_b*cell_X)/(old_b+cell_b));
	realtype new_Y = RCONST((old_b*old_Y+cell_b*cell_Y)/(old_b+cell_b));
	realtype new_A = RCONST((old_b*old_A+cell_b*cell_A)/(old_b+cell_b));
	realtype new_b = RCONST(old_b+cell_b);
	if(out==1)
		myfile_OC << "resulting cohort: " << new_m << " " << new_X << " " << new_Y << " " << new_A << " " << new_b << endl;
	cohnewit->set_init(new_m,new_X,new_Y,new_A,new_b); //change the values of the cohort at cohnewit to these new "merged values"
}

void addcohort_sub(clist &cohnew, clistit &cohnewit, realtype cell_m, realtype cell_X, realtype cell_Y, realtype cell_A, realtype cell_b, int out){
	/*adds cell_b cells (with cell_m, cell_X, cell_Y and cell_A) to a list of cohorts cohnew
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
		a.set_init(cell_m,cell_X,cell_Y,cell_A,cell_b);
		cohnew.push_back(a);
		cohnewit = cohnew.end();
		cohnewit--; //important for the search for the biggest parts (starts searching at cohnewit)
		if(out==1)
			myfile_OC << "-> bigger than the mass of every cohort in the list + delta, so create new cohort at the end of the list" << endl;
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
						//if mass-distance to the right cohort is smaller: add the cells to the right cohort (at location cohnewit+1)
						if(out==1)
							myfile_OC << "add cells to right cohort" << endl;
						cohnewit++;
						/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,X,Y,A*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out); 
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
						/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,X,Y,A*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out); 
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
					/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,X,Y,A*/
					merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out);
				}
			}
			else{ //the cells are in the inhibition zone of the last cohort in the list: add the cells to this cohort
				if(out==1)
					myfile_OC << "-> cells are in the inhibition zone of the last cohort in the list: merge the cells with this cohort" << endl;
				cohnewit--;
				/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the last cohort by using weighted means for m,X,Y,A*/
				merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out);
			}
		}
		else{ //add the cells in a new cohort
			if(out==1)
				myfile_OC << "-> the cells are not in the inhibition zone of a cohort of the list, so add a new cohort" << endl;
			a.set_init(cell_m,cell_X,cell_Y,cell_A,cell_b);
			cohnew.insert(cohnewit,a);
		}
	}
}

void addcohort(realtype p, clist &cohnew, realtype in, realtype m, realtype X, realtype Y, realtype A, int out){
	/*adds the daughter cells of "in" dividing cells (with m, X, Y and A) to a list of cohorts cohnew
	p is the parameter for mass division: daughter cells with p*m and with (1-p)*m are created
	whether the two resulting daughter cohorts are simply added to cohnew, or if they are merged with existing cohorts 
	      (by using weighted means) depends on the parameter delta: if the distance in mass to an existing cohort >= delta,
		  a new cohort is added, if the distance in mass is < delta, the daughter cohort is merged with the existing one
	parameter out (int): = 1 (give full output), = 0 (give only important output) */
	
	clistit cohnewit;
	cohort a;

	totaalb += 2*in; //increase the total amount of cells born during the age integration by 2 times the number of dividing cells

    /*determine cohort to which smallest parts after division contribute (so with p*m,X,Y,A,in)*/
	if(out==1)
		myfile_OC << "daughter cells with smallest mass" << endl;
	if(!cohnew.empty()){ //if cohnew is non-empty: search if there is cohort in cohnew for which p*m<(mass of cells in cohort + delta)
		cohnewit = cohnew.begin();
		addcohort_sub(cohnew,cohnewit,RCONST(p*m),X,Y,A,in,out);
	}
	else{ //if the list of cohorts is empty: created new cohort with p*m,X,Y,A,in
		if(out==1)
			myfile_OC << "the list of cohorts is empty -> create a new cohort with these cells" << endl;
		a.set_init(RCONST(p*m),X,Y,A,in);
		cohnew.push_back(a);
		cohnewit = cohnew.begin(); 
	}

	/*determine cohort to which biggest parts after division contribute (so with (1-p)*m,X,Y,A,in)*/
	if(out==1)
		myfile_OC << "daughter cells with biggest mass" << endl;
	// start search at cohnewit where the small parts are inserted
	addcohort_sub(cohnew,cohnewit,RCONST((1-p)*m),X,Y,A,in,out);
}

void integrate_allcohorts(clist coh, int out, int out2){ 
	/* corresponds to map M in paper 
	integrates a list of cohorts coh and calculates the adjustment to the nutrient 
	new birth cohorts in list cohnew, new S in tempS
	starting value of S is defined globally
	parameter out (int): = 1 (give full output), = 0 (give only important output) 
	parameter out2 : used to determine if matlab-file numberofcells.m has to be created 
		(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
		if == 0: no matlab-file is created, if == 1: the file is created*/
	
	realtype con = RCONST(0); //total amount of nutrient consumed by the cells in the cohorts in coh during the age integration
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
			if(cond_div2==1) //use special function integrate_cohort_bis if a probability distribution for X_div is used
				cohit->integrate_cohort_bis(out,out2); 
			else
				cohit->integrate_cohort(out,out2); 
			con += cohit->getb()*cohit->getTheta(); //calculate the amount of nutrient consumed by the cells in the cohort and add it to con 
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
				myfile_OC << "cohort " << p << " with initial mass " << cohit->getMass() << " X " << cohit->getX() << " Y " << cohit->getY() << " A " << cohit->getA() << " number " << cohit->getb() << endl;
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

int cohort::integrate_cohort_bis(int out, int out2) { 
	/* integrates the cohort until the age where F < Ft
	   if cohorts are created by division, they are added to the list cohnew (globally defined list)
	   when X is decreasing, the mass is in the correct range and X_div is reached 
	        (determined by a density probability for X_div and the use of a random number generator at certain age integration intervals
			(time in between determined by an exponential distribution)),
			-> all remaining cells in the cohort divide
	   parameter out (int): = 1 (give full output), = 0 (give only the most important output) 
	   parameter out2 : used to determine if matlab-file numberofcells.m has to be created 
			(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
			if == 0: no matlab-file is created, if == 1: the file is created*/

	realtype m, X, Y, A; /*the values of m,X,Y,A of the cells in the cohort at the current age integration output step*/
	realtype F_old, m_old, X_old, Y_old, A_old; /*the values of F,m,X,Y,A of the cells in the cohort at the previous age integration output step*/
	realtype Fwdiv; /*value of F when there's only decrease in F due to death during an age integration step*/
	int Fto0 = 0; /*if Fto0 is 1: the conditions are fulfilled that say that all remaining cells in the cohort should divide*/
	random_device rd;
    mt19937 gen(rd());
	exponential_distribution<> d(50); //asume that on the average the random generator is used ... times every unit of time/age 
	realtype a_rg = d(gen); //random number generated from an exponential distribution with parameter ... (50 here)
	uniform_real_distribution<> dis(0,1); //uniform real distribution in [0,1]
	realtype random_number;

	/*fill the matrix Xp with the values of the cumulative probability distribution for a certain number of X-values: 
	first row contains the X-values and the second row the corresponding cumulative probability
	ATTENTION: make sure that Xp[0][0]=100 and Xp[1][0]=0
	The first set of values is for normal distribution with mean 0.1 and sd 0.007
	The second set of values is for normal distribution with mean 0.1 and sd 0.02*/

	//alglib::real_2d_array Xp("[[100.0000000000,0.1460000000,0.1450000000,0.1440000000,0.1430000000,0.1420000000,0.1410000000,0.1400000000,0.1390000000,0.1380000000,0.1370000000,0.1360000000,0.1350000000,0.1340000000,0.1330000000,0.1320000000,0.1310000000,0.1300000000,0.1290000000,0.1280000000,0.1270000000,0.1260000000,0.1250000000,0.1240000000,0.1230000000,0.1220000000,0.1210000000,0.1200000000,0.1190000000,0.1180000000,0.1170000000,0.1160000000,0.1150000000,0.1140000000,0.1130000000,0.1120000000,0.1110000000,0.1100000000,0.1090000000,0.1080000000,0.1070000000,0.1060000000,0.1050000000,0.1040000000,0.1030000000,0.1020000000,0.1010000000,0.1000000000,0.0990000000,0.0980000000,0.0970000000,0.0960000000,0.0950000000,0.0940000000,0.0930000000,0.0920000000,0.0910000000,0.0900000000,0.0890000000,0.0880000000,0.0870000000,0.0860000000,0.0850000000,0.0840000000,0.0830000000,0.0820000000,0.0810000000,0.0800000000,0.0790000000,0.0780000000,0.0770000000,0.0760000000,0.0750000000,0.0740000000,0.0730000000,0.0720000000,0.0710000000,0.0700000000,0.0690000000,0.0680000000,0.0670000000,0.0660000000,0.0650000000,0.0640000000,0.0630000000,0.0620000000,0.0610000000,0.0600000000,0.0590000000,0.0580000000,0.0570000000,0.0560000000],[0.0000000000,0.0000000000,0.0000000001,0.0000000002,0.0000000004,0.0000000010,0.0000000024,0.0000000055,0.0000000126,0.0000000284,0.0000000626,0.0000001353,0.0000002867,0.0000005955,0.0000012128,0.0000024221,0.0000047430,0.0000091076,0.0000171503,0.0000316712,0.0000573601,0.0001018892,0.0001775197,0.0003033834,0.0005086207,0.0008365374,0.0013498980,0.0021373670,0.0033209427,0.0050639953,0.0075792194,0.0111354895,0.0160622856,0.0227501319,0.0316454161,0.0432381327,0.0580415669,0.0765637255,0.0992713968,0.1265489545,0.1586552539,0.1956829692,0.2375252620,0.2838545831,0.3341175709,0.3875484811,0.4432015032,0.5000000000,0.5567984968,0.6124515189,0.6658824291,0.7161454169,0.7624747380,0.8043170308,0.8413447461,0.8734510455,0.9007286032,0.9234362745,0.9419584331,0.9567618673,0.9683545839,0.9772498681,0.9839377144,0.9888645105,0.9924207806,0.9949360047,0.9966790573,0.9978626330,0.9986501020,0.9991634626,0.9994913793,0.9996966166,0.9998224803,0.9998981108,0.9999426399,0.9999683288,0.9999828497,0.9999908924,0.9999952570,0.9999975779,0.9999987872,0.9999994045,0.9999997133,0.9999998647,0.9999999374,0.9999999716,0.9999999874,0.9999999945,0.9999999976,0.9999999990,0.9999999996,0.9999999998]]"); //normal distribution with mean 0.1 and sd 0.007
	alglib::real_2d_array Xp("[[100.0000000000,0.2000000000,0.1990000000,0.1980000000,0.1970000000,0.1960000000,0.1950000000,0.1940000000,0.1930000000,0.1920000000,0.1910000000,0.1900000000,0.1890000000,0.1880000000,0.1870000000,0.1860000000,0.1850000000,0.1840000000,0.1830000000,0.1820000000,0.1810000000,0.1800000000,0.1790000000,0.1780000000,0.1770000000,0.1760000000,0.1750000000,0.1740000000,0.1730000000,0.1720000000,0.1710000000,0.1700000000,0.1690000000,0.1680000000,0.1670000000,0.1660000000,0.1650000000,0.1640000000,0.1630000000,0.1620000000,0.1610000000,0.1600000000,0.1590000000,0.1580000000,0.1570000000,0.1560000000,0.1550000000,0.1540000000,0.1530000000,0.1520000000,0.1510000000,0.1500000000,0.1490000000,0.1480000000,0.1470000000,0.1460000000,0.1450000000,0.1440000000,0.1430000000,0.1420000000,0.1410000000,0.1400000000,0.1390000000,0.1380000000,0.1370000000,0.1360000000,0.1350000000,0.1340000000,0.1330000000,0.1320000000,0.1310000000,0.1300000000,0.1290000000,0.1280000000,0.1270000000,0.1260000000,0.1250000000,0.1240000000,0.1230000000,0.1220000000,0.1210000000,0.1200000000,0.1190000000,0.1180000000,0.1170000000,0.1160000000,0.1150000000,0.1140000000,0.1130000000,0.1120000000,0.1110000000,0.1100000000,0.1090000000,0.1080000000,0.1070000000,0.1060000000,0.1050000000,0.1040000000,0.1030000000,0.1020000000,0.1010000000,0.1000000000,0.0990000000,0.0980000000,0.0970000000,0.0960000000,0.0950000000,0.0940000000,0.0930000000,0.0920000000,0.0910000000,0.0900000000,0.0890000000,0.0880000000,0.0870000000,0.0860000000,0.0850000000,0.0840000000,0.0830000000,0.0820000000,0.0810000000,0.0800000000,0.0790000000,0.0780000000,0.0770000000,0.0760000000,0.0750000000,0.0740000000,0.0730000000,0.0720000000,0.0710000000,0.0700000000,0.0690000000,0.0680000000,0.0670000000,0.0660000000,0.0650000000,0.0640000000,0.0630000000,0.0620000000,0.0610000000,0.0600000000,0.0590000000,0.0580000000,0.0570000000,0.0560000000,0.0550000000,0.0540000000,0.0530000000,0.0520000000,0.0510000000,0.0500000000,0.0490000000,0.0480000000,0.0470000000,0.0460000000,0.0450000000,0.0440000000,0.0430000000,0.0420000000,0.0410000000,0.0400000000,0.0390000000,0.0380000000,0.0370000000,0.0360000000,0.0350000000,0.0340000000,0.0330000000,0.0320000000,0.0310000000,0.0300000000,0.0290000000,0.0280000000,0.0270000000,0.0260000000,0.0250000000,0.0240000000,0.0230000000,0.0220000000,0.0210000000,0.0200000000,0.0190000000,0.0180000000,0.0170000000,0.0160000000,0.0150000000,0.0140000000,0.0130000000,0.0120000000,0.0110000000,0.0100000000,0.0090000000,0.0080000000,0.0070000000,0.0060000000,0.0050000000,0.0040000000,0.0030000000,0.0020000000,0.0010000000,0.0000000000],[0.0000000000,0.0000002867,0.0000003711,0.0000004792,0.0000006173,0.0000007933,0.0000010171,0.0000013008,0.0000016597,0.0000021125,0.0000026823,0.0000033977,0.0000042935,0.0000054125,0.0000068069,0.0000085399,0.0000106885,0.0000133457,0.0000166238,0.0000206575,0.0000256088,0.0000316712,0.0000390756,0.0000480963,0.0000590589,0.0000723480,0.0000884173,0.0001077997,0.0001311202,0.0001591086,0.0001926156,0.0002326291,0.0002802933,0.0003369293,0.0004040578,0.0004834241,0.0005770250,0.0006871379,0.0008163523,0.0009676032,0.0011442068,0.0013498980,0.0015888696,0.0018658133,0.0021859615,0.0025551303,0.0029797632,0.0034669738,0.0040245885,0.0046611880,0.0053861460,0.0062096653,0.0071428107,0.0081975359,0.0093867055,0.0107241100,0.0122244727,0.0139034475,0.0157776074,0.0178644206,0.0201822154,0.0227501319,0.0255880595,0.0287165598,0.0321567748,0.0359303191,0.0400591569,0.0445654628,0.0494714680,0.0547992917,0.0605707580,0.0668072013,0.0735292596,0.0807566592,0.0885079914,0.0968004846,0.1056497737,0.1150696702,0.1250719356,0.1356660609,0.1468590564,0.1586552539,0.1710561263,0.1840601253,0.1976625431,0.2118553986,0.2266273524,0.2419636522,0.2578461108,0.2742531178,0.2911596868,0.3085375387,0.3263552203,0.3445782584,0.3631693488,0.3820885778,0.4012936743,0.4207402906,0.4403823076,0.4601721627,0.4800611942,0.5000000000,0.5199388058,0.5398278373,0.5596176924,0.5792597094,0.5987063257,0.6179114222,0.6368306512,0.6554217416,0.6736447797,0.6914624613,0.7088403132,0.7257468822,0.7421538892,0.7580363478,0.7733726476,0.7881446014,0.8023374569,0.8159398747,0.8289438737,0.8413447461,0.8531409436,0.8643339391,0.8749280644,0.8849303298,0.8943502263,0.9031995154,0.9114920086,0.9192433408,0.9264707404,0.9331927987,0.9394292420,0.9452007083,0.9505285320,0.9554345372,0.9599408431,0.9640696809,0.9678432252,0.9712834402,0.9744119405,0.9772498681,0.9798177846,0.9821355794,0.9842223926,0.9860965525,0.9877755273,0.9892758900,0.9906132945,0.9918024641,0.9928571893,0.9937903347,0.9946138540,0.9953388120,0.9959754115,0.9965330262,0.9970202368,0.9974448697,0.9978140385,0.9981341867,0.9984111304,0.9986501020,0.9988557932,0.9990323968,0.9991836477,0.9993128621,0.9994229750,0.9995165759,0.9995959422,0.9996630707,0.9997197067,0.9997673709,0.9998073844,0.9998408914,0.9998688798,0.9998922003,0.9999115827,0.9999276520,0.9999409411,0.9999519037,0.9999609244,0.9999683288,0.9999743912,0.9999793425,0.9999833762,0.9999866543,0.9999893115,0.9999914601,0.9999931931,0.9999945875,0.9999957065,0.9999966023,0.9999973177,0.9999978875,0.9999983403,0.9999986992,0.9999989829,0.9999992067,0.9999993827,0.9999995208,0.9999996289,0.9999997133]]"); //normal distribution with mean 0.1 and sd 0.02
	int Xp_maxindex = Xp.cols()-1;
	int i = 0;

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
		myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
		   // the starting values of the cohort
	}

	if(out==1) //print the starting values of the cohort
		PrintOutput(iout,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
	realtype toutprev = RCONST(0); //save the previous output time in toutprev
	cvode_mem = CVodeCreate(CV_ADAMS,CV_FUNCTIONAL);
	if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);
	flag = CVodeInit(cvode_mem,f,toutprev,y);
	if (check_flag(&flag, "CVodeInit", 1)) return(1);
	flag = CVodeSStolerances(cvode_mem,reltol,abstol);
	if (check_flag(&flag, "CVodeSStolerances", 1)) return(1);
	/*change the maximum number of steps to be taken by the solver in its attempt to reach the next output time from its default value 500*/
	flag = CVodeSetMaxNumSteps(cvode_mem, 10000); 
	if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);

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


	while((Fto0!=1) //repeat age integration step on the cohort while conditions for division of the remaining cells are not fulfilled                 
			&& vwm_max0                      // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
			&& vwm_max1                      // and m<=m_max if cond_max=1 (always true otherwise)
			&& (NV_Ith_S(y,NDIM)>=Ft)){      // and survival probability of the cells in the cohort >= Ft

		/*save the values of F,m,X,Y,A of the cells in the cohort at the previous age integration output step*/
		F_old = NV_Ith_S(y,NDIM);
		m_old = NV_Ith_S(y,0);
		X_old = NV_Ith_S(y,1);
		Y_old = NV_Ith_S(y,2);
		A_old = NV_Ith_S(y,3);
		
		// take age integration step using the CVODE solver (without RootFinding)
		flag = CVode(cvode_mem,tout,y,&t,CV_NORMAL);

		if(out==1)
			PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
		if(out2==1)
			myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
		Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
													//ATTENTION: only valid when individual mortality function nu = D
		toutprev = t; //save current output time in toutprev 
		tout += 0.001;  //next output time

		/*update the booleans*/
		if(cond_min==1)
			vwm_min = (NV_Ith_S(y,0)>=m_min);
		if(cond_max==0)
			vwm_max0 = (NV_Ith_S(y,0)<=(m_max*(1-eps_m)));
		if(cond_max==1)
			vwm_max1 = (NV_Ith_S(y,0)<=m_max);

		//check if all remaining cells should divide (only do this when t>=a_rg and dX/dt<0 and and m >= m_min and m is not too big) 
		if((t>=a_rg)                                                 // if t>=a_rg (random number generator can be used to check if cells will divide)
				&& (k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1) < 0)   // and dX/dt<0
				&& vwm_min                                           // and m>=m_min
				&& vwm_max0                                          // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
				&& vwm_max1){                                        // and m<=m_max if cond_max=1 (always true otherwise)
			if(out==1){
				myfile_OC << "a_rg (" << a_rg << ") reached" << endl;
				myfile_OC << "Check if all remaining cells should divide" << endl;
			}
			//use random number generator to decide if all the remaining cells divide:
			while((i<=Xp_maxindex) && (NV_Ith_S(y,1)<=Xp[0][i])){
				i++;
			}
			i--; //i will be min. 1 since Xp[0][0]=100 (with Xp[1][0]=0)
			// => use cumulative probability at column i: Xp[1][i]
			random_number = dis(gen);
			if(out==1){
				myfile_OC << "Use cumulative probability at column " << i+1 << " with X_{DIV} equal to " << Xp[0][i] << " and cumulative probability to divide equal to " << Xp[1][i] << endl;
				myfile_OC << "Random number between 0 and 1: " << random_number << endl;
			}
			if(random_number<=Xp[1][i]){ //yes -> set Fto0 to 1 and let all the remaining cells divide
				Fto0 = 1;
				if(out==1){
					myfile_OC << "All remaining cells in the cohort divide" << endl;
					myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive and divide (from the original " << b << " cells in the cohort)"<< endl;
					myfile_OC << "number of cohorts in cohnew before division = " << cohnew.size() << endl; 
				}
				m = NV_Ith_S(y,0);
				X = NV_Ith_S(y,1);
				Y = NV_Ith_S(y,2);
				A = NV_Ith_S(y,3);
				theta = NV_Ith_S(y,NDIM+1);
				realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
				addcohort(phi,cohnew,in,m,X,Y,A,out);
				if(out==1){
					myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
					printListCoh_OC(cohnew);
				}
			}
			else{ //no -> adjust a_rg to next time when random number generator should be used to check if cells will divide
				a_rg = t+d(gen);
				if(out==1)
					myfile_OC << "no division, new a_rg = " << a_rg << endl;
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
	
	if(out2==1)
		myfile_out2.close(); 

	N_VDestroy_Serial(y);
	return(0);
}

void addcohort_dest_sub(clist &cohnew, clistit &cohnewit, int &i_cells, int i_start, int &how, realtype cell_m, realtype cell_X, realtype cell_Y, realtype cell_A, realtype cell_b, int out){
	/*adds cell_b cells (with cell_m, cell_X, cell_Y and cell_A) to a list of cohorts cohnew
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
		a.set_init(cell_m,cell_X,cell_Y,cell_A,cell_b);
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
						/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,X,Y,A*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out); 
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
						/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,X,Y,A*/
						merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out); 
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
					/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the cohort at cohnewit by using weighted means for m,X,Y,A*/
					merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out);
					how = 2;
				}
			}
			else{ //the cells are in the inhibition zone of the last cohort in the list: add the cells to this cohort
				if(out==1)
					myfile_OC << "-> cells are in the inhibition zone of the last cohort in the list: merge the cells with this cohort" << endl;
				cohnewit--;
				/*merge the new cells (cell_m,cell_X,cell_Y,cell_A,cell_b) with the cells in the last cohort by using weighted means for m,X,Y,A*/
				merge_Coh(cohnew,cohnewit,cell_m,cell_X,cell_Y,cell_A,cell_b,out);
				how = 2;
			}
		}
		else{ //add the cells in a new cohort
			if(out==1)
				myfile_OC << "-> the cells are not in the inhibition zone of a cohort of the list, so add a new cohort" << endl;
			a.set_init(cell_m,cell_X,cell_Y,cell_A,cell_b);
			cohnew.insert(cohnewit,a);
			cohnewit--; 
			how = 1;
			i_cells--;
		}
	}
}

void addcohort_dest(realtype p, clist &cohnew, realtype in, realtype m, realtype X, realtype Y, realtype A, int out, int &i_sm, int &how_sm, int &i_gr, int &how_gr){
	/*adds the daughter cells of "in" dividing cells (with m, X, Y and A) to a list of cohorts cohnew
	p is the parameter for mass division: daughter cells with p*m and with (1-p)*m are created
	whether the two resulting daughter cohorts are simply added to cohnew, or if they are merged with existing cohorts 
	      (by using weighted means) depends on the parameter delta: if the distance in mass to an existing cohort >= delta,
		  a new cohort is added, if the distance in mass is < delta, the daughter cohort is merged with the existing one
    see global definition for i_sm, how_sm, i_gr and how_gr -> contains the information needed to print data_figd3_fp.txt in integrate_allcohorts_dest
	parameter out (int): = 1 (give full output), = 0 (give only important output) */
	
	clistit cohnewit;
	cohort a;

	totaalb += 2*in; //increase the total amount of cells born during the age integration by 2 times the number of dividing cells

    /*determine cohort to which smallest parts after division contribute (so with p*m,X,Y,A,in)*/
	if(out==1)
		myfile_OC << "daughter cells with smallest mass" << endl;
	if(!cohnew.empty()){ //if cohnew is non-empty: search if there is cohort in cohnew for which p*m<(mass of cells in cohort + delta)
		cohnewit = cohnew.begin();
		addcohort_dest_sub(cohnew,cohnewit,i_sm,1,how_sm,RCONST(p*m),X,Y,A,in,out);
	}
	else{ //if the list of cohorts is empty: created new cohort with p*m,X,Y,A,in
		if(out==1)
			myfile_OC << "the list of cohorts is empty -> create a new cohort with these cells" << endl;
		a.set_init(RCONST(p*m),X,Y,A,in);
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

	/*determine cohort to which biggest parts after division contribute (so with (1-p)*m,X,Y,A,in)*/
	if(out==1)
		myfile_OC << "daughter cells with biggest mass" << endl;
	//start search at cohnewit where the small parts are inserted
	addcohort_dest_sub(cohnew,cohnewit,i_gr,i_start,how_gr,RCONST((1-p)*m),X,Y,A,in,out);
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

	realtype m, X, Y, A; /*the values of m,X,Y,A of the cells in the cohort at the current age integration output step*/
	realtype F_old, m_old, X_old, Y_old, A_old; /*the values of F,m,X,Y,A of the cells in the cohort at the previous age integration output step*/
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
		myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
		//the starting values of the cohort
	}

	if(out==1) //print the starting values of the cohort
		PrintOutput(iout,NV_Ith_S(y,0),NV_Ith_S(y,1),NV_Ith_S(y,2),NV_Ith_S(y,3),NV_Ith_S(y,NDIM),NV_Ith_S(y,NDIM+1));
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
	
	if(((NV_Ith_S(y,1))>=(X_div-1.0e-6))                     //all cells should divide immediately when: 
			&& ((NV_Ith_S(y,1))<=(X_div+1.0e-6))             // X in [X_div-1.0e-6,X_div+1.0e-6]
			&& k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1)<0   // and dX/dt<0
			&& vwm_min                                       // and m>=m_min if cond_min=1 (always true otherwise)
			&& vwm_max1){                                    // and m<=m_max if cond_max=1 (always true otherwise)
		if(out==1){
			myfile_OC << "Conditions for division are immediately fulfilled: all cells divide immediately" << endl;
			myfile_OC << NV_Ith_S(y,NDIM)*b << " cells are still alive (from the original " << b << " cells in the cohort)" << endl;
		}
		m = NV_Ith_S(y,0);
		X = NV_Ith_S(y,1);
		Y = NV_Ith_S(y,2);
		A = NV_Ith_S(y,3);
		theta = NV_Ith_S(y,NDIM+1); /*amount of nutrient consumed by one cell in the cohort up to its current age (when it divides)*/
		realtype in = NV_Ith_S(y,NDIM)*b; /*the number of cells (or concentration) that divide*/
		addcohort_dest(phi,cohnew,in,m,X,Y,A,out,i_sm,how_sm,i_gr,how_gr); 
		                            /*let the cells divide: add the resulting cohorts to the list cohnew*/
		if(out==1){
			myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
			printListCoh_OC(cohnew);
		}
	}
	else{ // when cells in the cohort don't all divide immediately
		while((Fto0!=1) //repeat age integration step on the cohort while: conditions for division of the remaining cells are not fulfilled                 
				&& vwm_max0                     // and m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
				&& vwm_max1                     // and m<=m_max if cond_max=1 (always true otherwise)
				&& (NV_Ith_S(y,NDIM)>=Ft)){     // and survival probability of the cells in the cohort >= Ft
			/*save the values of F,m,X,Y,A of the cells in the cohort at the previous age integration output step*/
			F_old = NV_Ith_S(y,NDIM);
			m_old = NV_Ith_S(y,0);
			X_old = NV_Ith_S(y,1);
			Y_old = NV_Ith_S(y,2);
			A_old = NV_Ith_S(y,3);
			
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
						PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
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
						//let these cells (with state values m_old,X_old,Y_old,A_old) divide: add the resulting cohorts to the list cohnew
						addcohort_dest(phi,cohnew,in,m_old,X_old,Y_old,A_old,out,i_sm,how_sm,i_gr,how_gr); 
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}

					//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
					if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                      // if X in [X_div-1.0e-6,X_div+1.0e-6]
							&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
							&& (k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1) < 0)  // and dX/dt<0
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
						X = NV_Ith_S(y,1);
						Y = NV_Ith_S(y,2);
						A = NV_Ith_S(y,3);
						theta = NV_Ith_S(y,NDIM+1);
						realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
						addcohort_dest(phi,cohnew,in,m,X,Y,A,out,i_sm,how_sm,i_gr,how_gr);
						if(out==1){
							myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
							printListCoh_OC(cohnew);
						}
					}
					F_old = NV_Ith_S(y,NDIM);
					m_old = NV_Ith_S(y,0);
					X_old = NV_Ith_S(y,1);
					Y_old = NV_Ith_S(y,2);
					A_old = NV_Ith_S(y,3);

					CVodeFree(&cvode_mem);
				}

				if((Fto0!=1) //do second age integration step on the cohort if conditions for division of the remaining cells are not fulfilled                 
					&& vwm_max0                    // and  m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
					&& vwm_max1                    // and  m<=m_max if cond_max=1 (always true otherwise)
					&& (NV_Ith_S(y,NDIM)>=Ft)){    // and  survival probability of the cells in the cohort >= Ft
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
					&& vwm_max0                          // and  m<=(m_max*(1-eps_m)) if cond_max=0 (always true otherwise)
					&& vwm_max1                         // and  m<=m_max if cond_max=1 (always true otherwise)
					&& (NV_Ith_S(y,NDIM)>=Ft)){         // and  survival probability of the cells in the cohort >= Ft
				flag = CVodeRootInit(cvode_mem, 1, g);
				if (check_flag(&flag, "CVodeRootInit", 1)) return(1);
				flag = CVode(cvode_mem,tout,y,&t,CV_NORMAL); //take age integration step using the CVODE solver	

				if(out==1)
					PrintOutput(t,NV_Ith_S(y,0),NV_Ith_S(y,1), NV_Ith_S(y,2), NV_Ith_S(y,3), NV_Ith_S(y,NDIM), NV_Ith_S(y,NDIM+1));
				if(out2==1)
					myfile_out2 << NV_Ith_S(y,0) << " " << NV_Ith_S(y,NDIM) << " " << RCONST(NV_Ith_S(y,NDIM)*b) << "; "; 
				Fwdiv = RCONST(F_old*exp(-D*(t-toutprev))); //value of F when there's only decrease in F due to death during this age integration step
															//ATTENTION: only valid when individual mortality function nu = D
				toutprev = t; //save current output time in toutprev 
				tout += TIN; //next output time

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
					//let these cells (with state values m_old,X_old,Y_old,A_old) divide: add the resulting cohorts to the list cohnew
					addcohort_dest(phi,cohnew,in,m_old,X_old,Y_old,A_old,out,i_sm,how_sm,i_gr,how_gr); 
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}

				//if all remaining cells should divide, set Fto0 to 1 and let all the remaining cells divide
				if(((NV_Ith_S(y,1)) >= (X_div-1.0e-6))                      // if X in [X_div-1.0e-6,X_div+1.0e-6]
						&& ((NV_Ith_S(y,1)) <= (X_div+1.0e-6)) 
						&& (k1-(k2p+k2pp*NV_Ith_S(y,2))*NV_Ith_S(y,1) < 0)  // and dX/dt<0
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
					X = NV_Ith_S(y,1);
					Y = NV_Ith_S(y,2);
					A = NV_Ith_S(y,3);
					theta = NV_Ith_S(y,NDIM+1);
					realtype in = RCONST(NV_Ith_S(y,NDIM)*b);
					addcohort_dest(phi,cohnew,in,m,X,Y,A,out,i_sm,how_sm,i_gr,how_gr);
					if(out==1){
						myfile_OC << "number of cohorts in cohnew after division = " << cohnew.size() << endl; 
						printListCoh_OC(cohnew);
					}
				}
			}
		}

		if((Fto0==1) && (out==1)) //if Fto0=1: all cells should have divided
			myfile_OC << "all cells should have divided" << endl; 
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
				myfile_OC << "cohort " << p << " with initial mass " << cohit->getMass() << " X " << cohit->getX() << " Y " << cohit->getY() << " A " << cohit->getA() << " number " << cohit->getb() << endl;
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

	/*print the output-file data_figd3_fp.txt*/

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
	/* k repeated evaluations of the map with adjustement of S and coh after each iteration
	   results: adjusted list of cohorts in coh and adjusted Sp in S (S globally defined)
	   parameter out (int): = 1 (full output), = 0 (only important output)
	   parameter file_numberofcells (int): = 1 (create a matlab-file numberofcells.m for the last iteration of the map with
	                                            for every cohort a matrix with the columns m; surv. prob.; number of cells alive) 
										   = 0 (creates no such file) */

	S = Sp; //set S to given value Sp
	int out2 = 0; //out2 is used to determine if matlab-file numberofcells.m has to be created 
	              //(a Matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
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
		                                                the new value of S, for every cohort m X Y A b, total amount of cells created*/
		myfile.open ("data.txt", ios::app);
		myfile << j << ", " << cohnew.size() << "," << tempS << ",";
		for(cohit=cohnew.begin();cohit!=cohnew.end();++cohit)
			myfile << cohit->getMass() << "," << cohit->getX() << "," << cohit->getY() << "," << cohit->getA() << "," << cohit->getb() << ",";
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
	   parameter out (int): = 1 (full output), = 0 (only important output) 
	   parameter file_numberofcells (int): = 1 (create a matlab-file numberofcells.m for the last iteration of the map with
	                                            for every cohort a matrix with the columns m; surv. prob.; number of cells alive) 
										   = 0 (creates no such file) */

	S = Sp; //set S to given value Sp
	int out2 = 0; //out2 is used to determine if matlab-file numberofcells.m has to be created 
	              //(a matlab-file that creates for every cohort a matrix with the columns m; surv. prob.; number of cells alive)
	              //if == 0: no matlab-file is created, if == 1: the file is created
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
		                                            the new value of S, for every cohort m X Y A b, total amount of cells created*/
	myfile.open ("data.txt", ios::app);
	myfile << 1 << ", " << cohnew.size() << "," << tempS << ",";
	for(cohit=cohnew.begin();cohit!=cohnew.end();++cohit)
		myfile << cohit->getMass() << "," << cohit->getX() << "," << cohit->getY() << "," << cohit->getA() << "," << cohit->getb() << ",";
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
		                                                the new value of S, for every cohort m X Y A b, total amount of cells created*/
		myfile.open ("data.txt", ios::app);
		myfile << i+1 << ", " << cohnew.size() << "," << tempS << ",";
		for(cohit=cohnew.begin();cohit!=cohnew.end();++cohit)
			myfile << cohit->getMass() << "," << cohit->getX() << "," << cohit->getY() << "," << cohit->getA() << "," << cohit->getb() << ",";
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

int contfpS0_simple(realtype &Sa, realtype &S0a, clist &coh, realtype S0d, realtype step_start, realtype step_m, realtype step_M, int k, realtype reltol_S, realtype reltol_btot, int out){
	/* continuation of the given fixed point by starting from the previous found fixed point
	start values of fixed point (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + S0)
	parameter S0d (realtype): continuation wanted until this value of S0
	parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-001)
	parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-005)
	parameter step_M: maximum value of the stepsize of the continuation (eg 1.0)
	parameter k, reltol_S and reltol_btot: ~ the convergence criteria, see loopMap_C
	parameter out (int): =1 (yes, give full output), = 0 (no, give only important output about the continuation)
	returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	changes to original values of Sa, S0a and coh made in function!*/

	ofstream myfile_summary; //file with summary of the fixed points: S0 N S btot
	myfile_summary.open("cont_summary.txt", ios::app); 
	myfile_summary.precision(10);

	ofstream myfile_fp; //file with the fixed points: S0 N S m1 X1 Y1 A1 b1 m2 ...
	myfile_fp.open("cont_fp.txt", ios::app); 
	myfile_fp.precision(10);

	ofstream myfile_full;
	if(out==1){ //file with more detailed information about the continuation
		myfile_full.open("cont_full.txt", ios::app); 
		myfile_full.precision(10);
	}

	if(S0d==S0a){ 
		if(out==1)
			myfile_full << "No fixed points calculated since the fixed point for the requested value of S0 is already given" << endl;
		return 1;
	}

	realtype step = step_start; //stepsize of the continuation, start at the given start value
	if(out==1)
		myfile_full << "step size = " << step << endl;
	if(S0d<S0a){
		step = -step_start; //S0 should be decreasing in the continuation
		if(out==1)
			myfile_full << "Start value of S0 = " << S0a << ", end value of S0 = " << S0d << ", so the step size is negative and equal to " << step << endl;
	}

	clist coh_calc; //used to save the list of cohorts of the previous fixed point 
	coh_calc.assign(coh.begin(),coh.end());
		
	/*print information about the fixed point at the start of the continuation*/
	clistit cohit = coh.begin();
	realtype btot = RCONST(0);
	myfile_summary << S0a << " " << coh.size() << " " << Sa << " "; 
	myfile_fp << S0a << " " << coh.size() << " " << Sa << " ";
	if(out==1)
		myfile_full << " Start fixed point with S0 " << S0a << ", S " << Sa << ", " << coh.size() << " cohorts with:" << endl;
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		btot += cohit->getb();
		myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	myfile_summary << btot << endl;
	myfile_fp << endl;
	if(out==1)
		myfile_full << endl;

	int max = k; //maximum number of iterations of the map for convergence
	S0a += step;

	while( ((step>0)&&((S0a-step)<S0d)) || ((step<0)&&((S0a-step)>S0d)) ){
		S0 = S0a;
		loopMap_C(Sa,coh,k,reltol_S,reltol_btot,0,0); //attention: value of k is adjusted to the number of times to map is effectively iterated
		if(out==1)
			myfile_full << k << " map iterations were needed to obtain the fixed point for S0 = " << S0a << endl;
		if(k==max){
			S0a = S0a-step;
			step = RCONST(step/2.0);
			S0a += step;
			coh.assign(coh_calc.begin(),coh_calc.end()); //start again from the last found fixed point
			if(out==1){
				myfile_full << "step size divided by 2 and now equal to " << step << endl;
				myfile_full << "repeat calculations for S0 = " << S0a << endl;
			}
			if(abs(step)<step_m){
				if(out==1)
					myfile_full << "step size smaller than " << step_m << ", stop continuation" << endl;
				return 1;
			}
		}

		else{
			if(k<50){
				step = RCONST(step*1.3);
				if(out==1)
					myfile_full << "step size multiplied by 1.3 and now equal to " << step << endl;
				if(abs(step)>step_M){
					if(step>0)
						step = step_M;
					else
						step = -step_M;
					if(out==1)
						myfile_full << "step size in absolute value bigger than " << step_M << ", so limit it to " << step << endl;
				}
			}

			/*print information about the new fixed point*/
			cohit = coh.begin();
			btot = RCONST(0);
			myfile_summary << S0a << " " << coh.size() << " " << S << " "; 
			myfile_fp << S0a << " " << coh.size() << " " << S << " ";
			if(out==1)
				myfile_full << " Fixed point found with S0 " << S0a << ", S " << S << ", " << coh.size() << " cohorts with:" << endl;
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				btot += cohit->getb();
				myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			myfile_summary << btot << endl;
			myfile_fp << endl;
			if(out==1)
				myfile_full << endl;

			Sa = S;
			S0a += step; 
			if(out==1)
				myfile_full << "calculations for S0 = " << S0a << endl;

			coh_calc.assign(coh.begin(),coh.end());
		}
		
		k = max; // k is reset to the original value

	}

	myfile_fp.close();
	myfile_summary.close();
	if(out==1)
		myfile_full.close();

	return 0;
}

int contfpD_simple(realtype &Sa, realtype &Da, clist &coh, realtype Dd, realtype step_start, realtype step_m, realtype step_M, int k, realtype reltol_S, realtype reltol_btot, int out){
	/* continuation of the given fixed point by starting from the previous found fixed point
	start values of fixed point (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + D)
	parameter Dd (realtype): continuation wanted until this value of D
	parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-003)
	parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-007)
	parameter step_M: maximum value of the stepsize of the continuation (eg 1.0e-002)
	parameter k, reltol_S and reltol_btot: ~ the convergence criteria, see loopMap_C
	parameter out (int): =1 (yes, give full output), = 0 (no, give only important output about the continuation)
	returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	changes to original values of Sa, Da and coh made in function!*/

	ofstream myfile_summary; //file with summary of the fixed points: D N S btot
	myfile_summary.open("cont_summary.txt", ios::app); 
	myfile_summary.precision(10);

	ofstream myfile_fp; //file with the fixed points: D N S m1 X1 Y1 A1 b1 m2 ...
	myfile_fp.open("cont_fp.txt", ios::app); 
	myfile_fp.precision(10);

	ofstream myfile_full;
	if(out==1){ //file with more information about the continuation
		myfile_full.open("cont_full.txt", ios::app); 
		myfile_full.precision(10);
	}

	if(Dd==Da){ 
		if(out==1)
			myfile_full << "No fixed points calculated since the fixed point for the requested value of D is already given" << endl;
		return 1;
	}

	realtype step = step_start; //stepsize of the continuation, start at the given start value
	if(out==1)
		myfile_full << "step size = " << step << endl;
	if(Dd<Da){
		step = -step_start; //D should be decreasing in the continuation
		if(out==1)
			myfile_full << "Start value of D = " << Da << ", end value of D = " << Dd << ", so the step size is negative and equal to " << step << endl;
	}

	clist coh_calc; //used to save the list of cohorts of the previous fixed point 
	coh_calc.assign(coh.begin(),coh.end());
		
	/*print information about the fixed point at the start of the continuation*/
	clistit cohit = coh.begin();
	realtype btot = RCONST(0);
	myfile_summary << Da << " " << coh.size() << " " << Sa << " "; 
	myfile_fp << Da << " " << coh.size() << " " << Sa << " ";
	if(out==1)
		myfile_full << " Start fixed point with D " << Da << ", S " << Sa << ", " << coh.size() << " cohorts with:" << endl;
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		btot += cohit->getb();
		myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	myfile_summary << btot << endl;
	myfile_fp << endl;
	if(out==1)
		myfile_full << endl;

	int max = k; //maximum number of iterations of the map for convergence
	Da += step;

	while( ((step>0)&&((Da-step)<Dd)) || ((step<0)&&((Da-step)>Dd)) ){
		D = Da;
		loopMap_C(Sa,coh,k,reltol_S,reltol_btot,0,0); //value of k is adjusted to the number of times to map is effectively iterated
		if(out==1)
			myfile_full << k << " map iterations were needed to obtain the fixed point for D = " << Da << endl;
		if(k==max){
			Da = Da-step;
			step = RCONST(step/2.0);
			Da += step;
			coh.assign(coh_calc.begin(),coh_calc.end()); //start again from the last found fixed point
			if(out==1){
				myfile_full << "step size divided by 2 and now equal to " << step << endl;
				myfile_full << "repeat calculations for D = " << Da << endl;
			}
			if(abs(step)<step_m){
				if(out==1)
					myfile_full << "step size smaller than " << step_m << ", stop continuation" << endl;
				return 1;
			}
		}
		else{
			if(k<50){
				step = RCONST(step*1.3);
				if(out==1)
					myfile_full << "step size multiplied by 1.3 and now equal to " << step << endl;
				if(abs(step)>step_M){
					if(step>0)
						step = step_M;
					else
						step = -step_M;
					if(out==1)
						myfile_full << "step size in absolute value bigger than " << step_M << ", so limit it to " << step << endl;
				}
			}
			/*print information about the new fixed point*/
			cohit = coh.begin();
			btot = RCONST(0);
			myfile_summary << Da << " " << coh.size() << " " << S << " "; 
			myfile_fp << Da << " " << coh.size() << " " << S << " ";
			if(out==1)
				myfile_full << " Fixed point found with D " << Da << ", S " << S << ", " << coh.size() << " cohorts with:" << endl;
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				btot += cohit->getb();
				myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			myfile_summary << btot << endl;
			myfile_fp << endl;
			if(out==1)
				myfile_full << endl;

			Sa = S;
			Da += step; 
			if(out==1)
				myfile_full << "calculations for D = " << Da << endl;

			coh_calc.assign(coh.begin(),coh.end());
		}
		k = max; // k is reset to the original value
	}

	myfile_fp.close();
	myfile_summary.close();
	if(out==1)
		myfile_full.close();

	return 0;
}

int contfpS0_2fp(realtype &S1, realtype &S01, clist &coh1, realtype &S2, realtype &S02, clist &coh2, realtype S0_d, realtype step_start, realtype step_m, realtype step_M, int k, realtype reltol_S, realtype reltol_btot, int out){
	/* continuation of the given fixed point by making an educated guess based on the 2 previous fixed points: 
	        the location of the cohorts of the last fixed point is used, S and btot is guessed based on linear extrapolation
	        and the number of cells in every cohort is rescaled (such that the total of cells is the new btot)
	values of the 2 last fixed points (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + S0) 
	ATTENTION: the 2 fixed points should be given in the right order
	parameter S0_d (realtype): continuation wanted until this value of S0
	parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-001)
	parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-005)
	parameter step_M: maximum value of the stepsize of the continuation (eg 1.0)
	parameter k, reltol_S and reltol_btot: ~ the convergence criteria, see loopMap_C
	parameter out (int): = 1 (yes, give full output), = 0 (no, give only important output about the continuation)
	returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	changes to original values of S1, S2, S01, S02, coh1 and coh2 made in function!*/

	ofstream myfile_summary; //file with summary of the fixed points: S0 N S btot
	myfile_summary.open("cont_summary.txt", ios::app); 
	myfile_summary.precision(10);

	ofstream myfile_fp; //file with the fixed points: S0 N S m1 X1 Y1 A1 b1 m2 ...
	myfile_fp.open("cont_fp.txt", ios::app); 
	myfile_fp.precision(10);

	ofstream myfile_full;
	if(out==1){ //file with more information about the continuation
		myfile_full.open("cont_full.txt", ios::app); 
		myfile_full.precision(10);
	}

	if(S0_d==S02){ 
		if(out==1)
			myfile_full << "No fixed points calculated since the fixed point for the requested value of S0 is already given" << endl;
		return 1;
	}

	realtype step = step_start; //stepsize of the continuation, start at the given start value
	if(out==1)
		myfile_full << "step size = " << step << endl;
	if(S0_d<S02){
		step = -step_start; //S0 should be decreasing in the continuation
		if(out==1)
			myfile_full << "Start value of S0 = " << S02 << ", end value of S0 = " << S0_d << ", so the step size is negative and equal to " << step << endl;
	}
		
	/*print information about the two fixed points at the start of the continuation*/
	clistit cohit = coh1.begin();
	realtype btot1 = RCONST(0);
	myfile_summary << S01 << " " << coh1.size() << " " << S1 << " "; 
	myfile_fp << S01 << " " << coh1.size() << " " << S1 << " ";
	if(out==1)
		myfile_full << " Start fixed point 1 with S0 " << S01 << ", S " << S1 << ", " << coh1.size() << " cohorts with:" << endl;
	for(cohit=coh1.begin();cohit!=coh1.end();++cohit){
		btot1 += cohit->getb();
		myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	myfile_summary << btot1 << endl;
	myfile_fp << endl;
	if(out==1){
		myfile_full << endl;
		myfile_full << "btot1 = " << btot1 << endl;
	}

	cohit = coh2.begin();
	realtype btot2 = RCONST(0);
	myfile_summary << S02 << " " << coh2.size() << " " << S2 << " "; 
	myfile_fp << S02 << " " << coh2.size() << " " << S2 << " ";
	if(out==1)
		myfile_full << " Start fixed point 2 with S0 " << S02 << ", S " << S2 << ", " << coh2.size() << " cohorts with:" << endl;
	for(cohit=coh2.begin();cohit!=coh2.end();++cohit){
		btot2 += cohit->getb();
		myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	myfile_summary << btot2 << endl;
	myfile_fp << endl;
	if(out==1){
		myfile_full << endl;
		myfile_full << "btot2 = " << btot2 << endl;
	}

	int max = k; //maximum number of iterations of the map for convergence

	//make prediction for the next fixed point: coh_new, S_new and S0_new
	realtype S0_new = S02;
	S0_new += step;
	if(out==1)
		myfile_full << "prediction for S0 = " << S0_new << endl;
	//define coh_new (and btot_new) and S_new
	realtype btot_new = RCONST(btot1+((S0_new-S01)/(S02-S01))*(btot2-btot1));
	if(out==1)
		myfile_full << "prediction for btot = " << btot_new << endl;
	if(btot_new<0){
		if(out==1)
			myfile_full << "btot_new is predicted negative, so change prediction to 0" << endl;
		btot_new = 0;
	}
	clist coh_new;
	coh_new.assign(coh2.begin(),coh2.end());
	realtype b;
	if(out==1)
		myfile_full << "prediction for list of cohorts:" << endl;
	for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
		b = RCONST(cohit->getb()*btot_new/btot2);
		cohit->setb(b);
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	realtype S_new = RCONST(S1+((S0_new-S01)/(S02-S01))*(S2-S1));
	if(out==1)
		myfile_full << "prediction for S = " << S_new << endl;
	if(S_new<0){
		if(out==1)
			myfile_full << "S_new is predicted negative, so change prediction to 0.000001" << endl;
		S_new = 0.000001;
	}

	while( ((step>0)&&((S0_new-step)<S0_d)) || ((step<0)&&((S0_new-step)>S0_d)) ){
		S0 = S0_new;
		loopMap_C(S_new,coh_new,k,reltol_S,reltol_btot,0,0); //value of k is adjusted to the number of times to map is effectively iterated
		if(out==1)
			myfile_full << k << " map iterations were needed to obtain the fixed point for S0 = " << S0_new << endl;
		if(k==max){
			S0_new = S0_new-step;
			step = RCONST(step/2.0);
			S0_new += step;
			//adjust the amount of cells in the cohorts of coh_new (and btot_new) and S_new
			btot_new = RCONST(btot1+((S0_new-S01)/(S02-S01))*(btot2-btot1));
			if(btot_new<0){
				if(out==1)
					myfile_full << "btot_new is predicted negative, so change prediction to 0" << endl;
				btot_new = 0;
			}
			coh_new.assign(coh2.begin(),coh2.end());
			if(out==1){
				myfile_full << "S01 = " << S01 << ", S02 = " << S02 << ", S0_new = " << S0_new << endl;
				myfile_full << "btot1 = " << btot1 << ", btot2 = " << btot2 << endl;
				myfile_full << "S1 = " << S1 << ", S2 = " << S2 << endl;
				myfile_full << "prediction for btot = " << btot_new << endl;
				myfile_full << "prediction for list of cohorts:" << endl;
			}
			for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
				b = RCONST(cohit->getb()*btot_new/btot2);
				cohit->setb(b);
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			S_new = RCONST(S1+((S0_new-S01)/(S02-S01))*(S2-S1));
			if(out==1)
				myfile_full << "prediction for S = " << S_new << endl;
			if(S_new<0){
				if(out==1)
					myfile_full << "S_new is predicted negative, so change prediction to 0.000001" << endl;
				S_new = 0.000001;
			}
			if(out==1){
				myfile_full << "step size divided by 2 and now equal to " << step << endl;
				myfile_full << "repeat calculations for S0 = " << S0_new << endl;
			}
			if(abs(step)<step_m){
				if(out==1)
					myfile_full << "step size smaller than " << step_m << ", stop continuation" << endl;
				return 1;
			}
		}

		else{
			if(k<50){
				step = RCONST(step*1.3);
				if(out==1)
					myfile_full << "step size multiplied by 1.3 and now equal to " << step << endl;
				if(abs(step)>step_M){
					if(step>0)
						step = step_M;
					else
						step = -step_M;
					if(out==1)
						myfile_full << "step size in absolute value bigger than " << step_M << ", so limit it to " << step << endl;
				}
			}

			/*print information about the new fixed point*/
			cohit = coh_new.begin();
			btot_new = RCONST(0);
			myfile_summary << S0_new << " " << coh_new.size() << " " << S << " "; 
			myfile_fp << S0_new << " " << coh_new.size() << " " << S << " ";
			if(out==1)
				myfile_full << " Fixed point found with S0 " << S0_new << ", S " << S << ", " << coh_new.size() << " cohorts with:" << endl;
			for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
				btot_new += cohit->getb();
				myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			myfile_summary << btot_new << endl;
			myfile_fp << endl;
			if(out==1)
				myfile_full << endl;

			S_new = S;

			//replace coh1 by coh2, S1 by S2, S01 by S02, btot1 by btot2
			coh1.assign(coh2.begin(),coh2.end());
			S1 = S2;
			S01 = S02;
			btot1 = btot2;
			//replace coh2 by coh_new, S2 by S_new, S02 by S0_new, btot2 by btot_new
			coh2.assign(coh_new.begin(),coh_new.end());
			S2 = S_new;
			S02 = S0_new;
			btot2 = btot_new;
			//make new prediction: change coh_new (and btot_new) and S_new 
			S0_new += step; 
			if(out==1)
				myfile_full << "prediction for S0 = " << S0_new << endl;
			coh_new.assign(coh2.begin(),coh2.end());
			btot_new = RCONST(btot1+((S0_new-S01)/(S02-S01))*(btot2-btot1));
			if(out==1)
				myfile_full << "prediction for btot = " << btot_new << endl;
			if(btot_new<0){
				if(out==1)
					myfile_full << "btot_new is predicted negative, so change prediction to 0" << endl;
				btot_new = 0;
			}
			if(out==1)
				myfile_full << "prediction for list of cohorts:" << endl;
			for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
				b = RCONST(cohit->getb()*btot_new/btot2);
				cohit->setb(b);
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			if(out==1)
				myfile_full << "S1 = " << S1 << ", S2 = " << S2 << ", S01 = " << S01 << ", S02 = " << S02 << endl;
			S_new = RCONST(S1+((S0_new-S01)/(S02-S01))*(S2-S1));
			if(out==1)
				myfile_full << "prediction for S = " << S_new << endl;
			if(S_new<0){
				if(out==1)
					myfile_full << "S_new is predicted negative, so change prediction to 0.000001" << endl;
				S_new = 0.000001;
			}
		}
		
		k = max; // k is reset to the original value
	}

	myfile_fp.close();
	myfile_summary.close();
	if(out==1)
		myfile_full.close();

	return 0;
}

int contfpD_2fp(realtype &S1, realtype &D1, clist &coh1, realtype &S2, realtype &D2, clist &coh2, realtype D_d, realtype step_start, realtype step_m, realtype step_M, int k, realtype reltol_S, realtype reltol_btot, int out){
	/* continuation of the given fixed point by making an educated guess based on the 2 previous fixed points: 
	     the location of the cohorts of the last fixed point is used, S and btot is guessed based on linear extrapolation
	     and the number of cells in every cohort is rescaled (such that the total of cells is new btot)
	values of the 2 last fixed points (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + D) 
	attention: the 2 fixed points should be given in the right order
	parameter D_d (realtype): continuation wanted until this value of D
	parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-001)
	parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-005)
	parameter step_M: maximum value of the stepsize of the continuation (eg 1.0)
	parameter k, reltol_S and reltol_btot: ~ the convergence criteria, see loopMap_C
	parameter out (int): =1 (yes, give full output), = 0 (no, give only important output about the continuation)
	returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	changes to original values of S1, S2, D1, D2, coh1 and coh2 made in function!*/

	ofstream myfile_summary; //file with summary of the fixed points: D N S btot
	myfile_summary.open("cont_summary.txt", ios::app); 
	myfile_summary.precision(10);

	ofstream myfile_fp; //file with the fixed points: D N S m1 X1 Y1 A1 b1 m2 ...
	myfile_fp.open("cont_fp.txt", ios::app); 
	myfile_fp.precision(10);

	ofstream myfile_full;
	if(out==1){ //file with more information about the continuation
		myfile_full.open("cont_full.txt", ios::app); 
		myfile_full.precision(10);
	}

	if(D_d==D2){ 
		if(out==1)
			myfile_full << "No fixed points calculated since the fixed point for the requested value of D is already given" << endl;
		return 1;
	}

	realtype step = step_start; //stepsize of the continuation, start at the given start value
	if(out==1)
		myfile_full << "step size = " << step << endl;
	if(D_d<D2){
		step = -step_start; //D should be decreasing in the continuation
		if(out==1)
			myfile_full << "Start value of D = " << D2 << ", end value of D = " << D_d << ", so the step size is negative and equal to " << step << endl;
	}
		
	/*print information about the two fixed points at the start of the continuation*/
	clistit cohit = coh1.begin();
	realtype btot1 = RCONST(0);
	myfile_summary << D1 << " " << coh1.size() << " " << S1 << " "; 
	myfile_fp << D1 << " " << coh1.size() << " " << S1 << " ";
	if(out==1)
		myfile_full << " Start fixed point 1 with D " << D1 << ", S " << S1 << ", " << coh1.size() << " cohorts with:" << endl;
	for(cohit=coh1.begin();cohit!=coh1.end();++cohit){
		btot1 += cohit->getb();
		myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	myfile_summary << btot1 << endl;
	myfile_fp << endl;
	if(out==1){
		myfile_full << endl;
		myfile_full << "btot1 = " << btot1 << endl;
	}

	cohit = coh2.begin();
	realtype btot2 = RCONST(0);
	myfile_summary << D2 << " " << coh2.size() << " " << S2 << " "; 
	myfile_fp << D2 << " " << coh2.size() << " " << S2 << " ";
	if(out==1)
		myfile_full << " Start fixed point 2 with D " << D2 << ", S " << S2 << ", " << coh2.size() << " cohorts with:" << endl;
	for(cohit=coh2.begin();cohit!=coh2.end();++cohit){
		btot2 += cohit->getb();
		myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	myfile_summary << btot2 << endl;
	myfile_fp << endl;
	if(out==1){
		myfile_full << endl;
		myfile_full << "btot2 = " << btot2 << endl;
	}

	int max = k; //maximum number of iterations of the map for convergence

	//make prediction for the next fixed point: coh_new, S_new and D_new
	realtype D_new = D2;
	D_new += step;
	if(out==1)
		myfile_full << "prediction for D = " << D_new << endl;
	//coh_new (and btot_new) and S_new definiëren
	realtype btot_new = RCONST(btot1+((D_new-D1)/(D2-D1))*(btot2-btot1));
	if(out==1)
		myfile_full << "prediction for btot = " << btot_new << endl;
	if(btot_new<0){
		if(out==1)
			myfile_full << "btot_new is predicted negative, so change prediction to 0" << endl;
		btot_new = 0;
	}
	clist coh_new;
	coh_new.assign(coh2.begin(),coh2.end());
	realtype b;
	if(out==1)
		myfile_full << "prediction for list of cohorts:" << endl;
	for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
		b = RCONST(cohit->getb()*btot_new/btot2);
		cohit->setb(b);
		if(out==1)
			myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
	realtype S_new = RCONST(S1+((D_new-D1)/(D2-D1))*(S2-S1));
	if(out==1)
		myfile_full << "prediction for S = " << S_new << endl;
	if(S_new<0){
		if(out==1)
			myfile_full << "S_new is predicted negative, so change prediction to 0.000001" << endl;
		S_new = 0;
	}

	while( ((step>0)&&((D_new-step)<D_d)) || ((step<0)&&((D_new-step)>D_d)) ){
		D = D_new;
		loopMap_C(S_new,coh_new,k,reltol_S,reltol_btot,0,0); //value of k is adjusted to the number of times to map is effectively iterated

		if(out==1)
			myfile_full << k << " map iterations were needed to obtain the fixed point for D = " << D_new << endl;

		if(k==max){
			D_new = D_new-step;
			step = RCONST(step/2.0);
			D_new += step;
			//adjust the amount of cells in the cohorts of coh_new (and btot_new) and S_new
			btot_new = RCONST(btot1+((D_new-D1)/(D2-D1))*(btot2-btot1));
			if(btot_new<0){
				if(out==1)
					myfile_full << "btot_new is predicted negative, so change prediction to 0" << endl;
				btot_new = 0;
			}
			coh_new.assign(coh2.begin(),coh2.end());
			if(out==1){
				myfile_full << "D1 = " << D1 << ", D2 = " << D2 << ", D_new = " << D_new << endl;
				myfile_full << "btot1 = " << btot1 << ", btot2 = " << btot2 << endl;
				myfile_full << "S1 = " << S1 << ", S2 = " << S2 << endl;
				myfile_full << "prediction for btot = " << btot_new << endl;
				myfile_full << "prediction for list of cohorts:" << endl;
			}
			for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
				b = RCONST(cohit->getb()*btot_new/btot2);
				cohit->setb(b);
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			S_new = RCONST(S1+((D_new-D1)/(D2-D1))*(S2-S1));
			if(out==1)
				myfile_full << "prediction for S = " << S_new << endl;
			if(S_new<0){
				if(out==1)
					myfile_full << "S_new is predicted negative, so change prediction to 0.000001" << endl;
				S_new = 0;
			}
			if(out==1){
				myfile_full << "step size divided by 2 and now equal to " << step << endl;
				myfile_full << "repeat calculations for D = " << D_new << endl;
			}
			if(abs(step)<step_m){
				if(out==1)
					myfile_full << "step size smaller than " << step_m << ", stop continuation" << endl;
				return 1;
			}
		}

		else{
			if(k<50){
				step = RCONST(step*1.3);
				if(out==1)
					myfile_full << "step size multiplied by 1.3 and now equal to " << step << endl;
				if(abs(step)>step_M){
					if(step>0)
						step = step_M;
					else
						step = -step_M;
					if(out==1)
						myfile_full << "step size in absolute value bigger than " << step_M << ", so limit it to " << step << endl;
				}
			}
			/*print information about the new fixed point*/
			cohit = coh_new.begin();
			btot_new = RCONST(0);
			myfile_summary << D_new << " " << coh_new.size() << " " << S << " "; 
			myfile_fp << D_new << " " << coh_new.size() << " " << S << " ";
			if(out==1)
				myfile_full << " Fixed point found with D " << D_new << ", S " << S << ", " << coh_new.size() << " cohorts with:" << endl;
			for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
				btot_new += cohit->getb();
				myfile_fp << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " ";
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			myfile_summary << btot_new << endl;
			myfile_fp << endl;
			if(out==1)
				myfile_full << endl;

			S_new = S;

			//replace coh1 by coh2, S1 by S2, D1 by D2, btot1 by btot2
			coh1.assign(coh2.begin(),coh2.end());
			S1 = S2;
			D1 = D2;
			btot1 = btot2;
			//replace coh2 by coh_new, S2 by S_new, D2 by D_new, btot2 by btot_new
			coh2.assign(coh_new.begin(),coh_new.end());
			S2 = S_new;
			D2 = D_new;
			btot2 = btot_new;
			//make new prediction: change coh_new (and btot_new) and S_new 
			D_new += step; 
			if(out==1)
				myfile_full << "prediction for D = " << D_new << endl;
			coh_new.assign(coh2.begin(),coh2.end());
			btot_new = RCONST(btot1+((D_new-D1)/(D2-D1))*(btot2-btot1));
			if(out==1)
				myfile_full << "prediction for btot = " << btot_new << endl;
			if(btot_new<0){
				if(out==1)
					myfile_full << "btot_new is predicted negative, so change prediction to 0" << endl;
				btot_new = 0;
			}
			if(out==1)
				myfile_full << "prediction for list of cohorts:" << endl;
			for(cohit=coh_new.begin();cohit!=coh_new.end();++cohit){
				b = RCONST(cohit->getb()*btot_new/btot2);
				cohit->setb(b);
				if(out==1)
					myfile_full << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
			if(out==1)
				myfile_full << "S1 = " << S1 << ", S2 = " << S2 << ", D1 = " << D1 << ", D2 = " << D2 << endl;
			S_new = RCONST(S1+((D_new-D1)/(D2-D1))*(S2-S1));
			if(out==1)
				myfile_full << "prediction for S = " << S_new << endl;
			if(S_new<0){
				if(out==1)
					myfile_full << "S_new is predicted negative, so change prediction to 0.000001" << endl;
				S_new = 0;
			}
		}
		k = max; // k is reset to the original value
	}

	myfile_fp.close();
	myfile_summary.close();
	if(out==1)
		myfile_full.close();

	return 0;
} 

void mapM_B(clist coh_new, clist coh_or, int out){ 
	/* map M_B: the mapping of a list of cohorts (coh_new) to a list of cohorts (coh_or) 
	   a new list of cohorts is created (cohadj) that is identical to coh_or, except for the number of cells (b) in the cohorts
	   --> new list of birth cohorts in cohadj (global)
	   we start with a list of cohorts cohadj that is identical to coh_or, but with all the cohorts empty
	   then the cells of each cohort in coh_new are allocated to the cohort of cohadj that is closest (in mass) to these cells */

	cohort a;
	clistit cohnewit = coh_new.begin();
	realtype in = RCONST(0); //number of cells in a cohort in the new list cohadj

	/*empty the list of cohorts cohadj*/
	clistit cohit = cohadj.begin();
	while(cohit!=cohadj.end())
		cohit = cohadj.erase(cohit);

	//save the values of the mass, X, Y and A of the first cohort in coh_or in m, X, Y and A
	cohit = coh_or.begin();
	realtype m = cohit->getMass();
	realtype X = cohit->getX();
	realtype Y = cohit->getY();
	realtype A = cohit->getA();

	cohit++; // go to the next cohort in coh_or
	while(cohit!=coh_or.end()){
		while((cohnewit!=coh_new.end())
			&& (cohnewit->getMass() <= RCONST((m+cohit->getMass())/2))){ 
		/*while the mass of the cohort at cohnewit is closer to the mass in the previous cohort of coh_or (cohit--)
		  than to the mass of the cohort at cohit: add the number of cells of the cohort in cohnewit to the previous cohort
		  --> add this number of cells to in */
			in += cohnewit->getb();
			cohnewit++;
		}
		a.set_init(m,X,Y,A,in); //create a new cohort in cohadj with m,X,Y,A (the values of the previous cohort) and in cells
		cohadj.push_back(a);
		//save the values of the mass, X, Y and A of the cohort at cohit in coh_or 
		m = cohit->getMass();
		X = cohit->getX();
		Y = cohit->getY();
		A = cohit->getA();
		cohit++; //go to the next cohort in coh_or
		in = RCONST(0); //set in back to 0
	}

	while(cohnewit!=coh_new.end()){ 
	//if there are cohorts in coh_new with mass bigger than the mass of the last cohort of coh_or
	// --> add the cells of these cohorts to the cohort in cohadj corresponding to the last cohort of coh_or (values m,X,Y,A)
		in += cohnewit->getb();
		cohnewit++;
	}
	a.set_init(m,X,Y,A,in); 
	cohadj.push_back(a);
}

void mapM_1(clist coh, int out){ 
	/* map M_1 used for continuation of fixed points: integrate_allcohorts on coh + M_B on cohnew (back to the original list of cohorts)
	   new list of birth cohorts in cohadj (global), new S in tempS (global)
	   parameter out (int): = 1 (full output), = 0 (only important output) */

	integrate_allcohorts(coh,out,0); //creates tempS (global) and new list of cohorts cohnew (global)
	mapM_B(cohnew,coh,out); //creates new list of cohorts cohadj (same m,X,Y,A as the cohorts in coh) that corresponds to cohnew

	ofstream myfile;
	myfile.precision(10);
	myfile.open ("mapM1.txt", ios::app); 
	for(clistit cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
	}
}

alglib::real_1d_array normalize(alglib::real_1d_array a){ //function that normalizes a given real_1d_array a
	alglib::real_1d_array an;
	an.setlength(a.length());
	realtype normsq = RCONST(0.0);
	for(int i=0;i<a.length();++i)
		normsq += a(i)*a(i);
	realtype norm = RCONST(sqrt(normsq));
	for(int i=0;i<a.length();++i)
		an(i) = RCONST(a(i)/norm);
	return an;
}

int signdotproduct(alglib::real_1d_array voud, alglib::real_1d_array vnew, alglib::real_1d_array moud, alglib::real_1d_array mnew, int out){ 
	/*calculates approximation of dotproduct of vectors voud and vnew (with each corresponding mass distribution moud and mnew) and returns the sign
	  length of v-vector should be length of m-vector+2 since v=(S b(coh1) ... b(cohN) S0) and m=(m(coh1) ... m(cohN))
	  function exists because the number of cohorts (thus also the length of the birth vector) can be different for the 2 given vectors
	  if the number of cohorts is the same: the summation of boud(coh i)*bnew(coh i) over the cohorts is calculated and the sign is determined
	  returns +1 if sign of dotproduct is + and -1 if sign is -
	  returns 0 if sizes of vectors are incorrect
	  parameter out (int): = 1 (give full output), = 0 (only give important output)*/

	ofstream myfile;
	myfile.open("signdotproduct.txt", ios::app); 
	myfile.precision(10);
	if(out==1)
		myfile << "length voud = " << voud.length() << ", length moud = " << moud.length() << ", length vnew = " << vnew.length() << ", length mnew = " << mnew.length() << endl;
	if(((voud.length()-2)!=moud.length()) //length of v-vector should be length of m-vector+2 since v=(S b(coh1) ... b(cohN) S0) and m=(m(coh1) ... m(cohN))
		|| ((vnew.length()-2)!=mnew.length())){
		if(out==1){
			myfile << "lengths of given vectors are incorrect" << endl;
			myfile << endl;
		}
		return 0;
	}
	else{
		realtype product = RCONST(0);
		realtype diff;
		int k = 0; //for moud
		int l = 0; //for mnew
		if(moud.length()<=mnew.length()){
			//use the values of moud (=> voud) and use the closest values in mnew (=> vnew) as approximation
			if(out==1)
				myfile << "use values of moud (=> voud) and use closest values in mnew (=> vnew) as approximation" << endl;
			for(k=0;k<moud.length();++k){ //search "closest" l
				if(out==1)
					myfile << "k = " << k << endl;
				diff = RCONST(abs(moud(k)-mnew(l)));
				while((abs(moud(k)-mnew(l))<=diff) && (l<mnew.length())){
					diff = RCONST(abs(moud(k)-mnew(l)));
					if(out==1)
						myfile << "l = " << l << " diff = " << diff << ", ";
					l++;
				}
				if(out==1)
					myfile << "l = " << l << " " << abs(moud(k)-mnew(l)) << ", ";
				l--;
				if(out==1)
					myfile << "use l = " << l << endl;
				product += RCONST(voud(k+1)*vnew(l+1));
				if(out==1)
					myfile << "product is now = " << product << " (" << RCONST(voud(k+1)*vnew(l+1)) << " added)" << endl;
			}
		}
		else{
			//use values of mnew (=>vnew) and use closest values in moud (=>voud) as approximation
			if(out==1)
				myfile << "use values of mnew (=>vnew) and use closest values in moud (=>voud) as approximation" << endl;
			for(l=0;l<mnew.length();++l){ //search "closest" k
				if(out==1)
					myfile << "l = " << l << endl;
				diff = RCONST(abs(moud(k)-mnew(l)));
				while((abs(moud(k)-mnew(l))<=diff) && (k<moud.length())){
					diff = RCONST(abs(moud(k)-mnew(l)));
					if(out==1)
						myfile << "k = " << k << " diff = " << diff << ", ";
					k++;
				}
				if(out==1)
					myfile << "k = " << k << " " << abs(moud(k)-mnew(l)) << ", ";
				k--;
				if(out==1)
					myfile << "use k = " << k << endl;
				product += RCONST(voud(k+1)*vnew(l+1));
				if(out==1)
					myfile << "product is now = " << product << " (" << RCONST(voud(k+1)*vnew(l+1)) << " added)" << endl;
			}
		}

		if(product>=0){
			if(out==1){
				myfile << "product>=0, so return 1" << endl;
				myfile << endl;
			}
			return 1;
		}
		else{
			if(out==1){
				myfile << "product<0, so return -1" << endl;
				myfile << endl;
			}
			return -1;
		}
	}
}

alglib::real_2d_array JacobianGS0(realtype Sa, realtype S0a, clist coh, int out){ 
	/* calculates the Jacobian of the map G := mapM_1 - identity (has to be zero for a fixed point)
	   in the point S=Sa, S0=S0a and list of cohorts = coh (coh => number of cells AND the mass-distribution)
	   parameter out (int): = 1 (give full output), = 0 (only give important output)*/

	S0 = S0a;

	alglib::real_2d_array A; //the Jacobian that will be returned
	int n1 = coh.size()+1; //number of rows of the Jacobian
	int n2 = coh.size()+2; //number of columns of the Jacobian
	A.setlength(n1,n2);

	alglib::real_2d_array Sb; //matrix with the results of the calculations for the Jacobian 
	int n3 = 2*coh.size()+4; //number of columns = how many times the map has to be evaluated (1 column = 1 map evaluation)
	Sb.setlength(n1,n3); //number of rows = 1 (S) + number of cohorts (b) 

	ofstream myfile2;
	myfile2.precision(10);
	myfile2.open("mapM1.txt", ios::app); 
	
	realtype eps_S0 = RCONST(0.01*S0a); //eps_S0 = 1% of S0
	realtype eps_S = RCONST(0.01*Sa); //eps_S = 1% of S
	realtype epsilon_b = RCONST(0.01); //epsilon_b = percentage of each b that will be used for eps_B = 1%
	realtype eps_B; //value of epsilon used for calculation of the derivative with respect to a certain b
	                //calculated as epsilon_b (originally 1%) * b for each b  
	                //attention: not the same as eps_b (see constants definitions)
	alglib::real_1d_array eps_B_vec; //vector where all the used eps_B are stored (length = number of cohorts)
	eps_B_vec.setlength(coh.size());
	
	clist coht = coh; //used for the map evalution where one of the b-values of coh is adjusted (for the derivatives with respect to b)
	                  //initially coht = coh
	clistit cohit;
	clistit cohadjit;
	clistit cohtit;

	int i; //used for the iteration over the rows of the matrix Sb

	/*M1 for S=Sa+eps_S and original number of births in the cohorts and S0=S0a*/
	if(out==1){
		myfile_OC << "step 1 (S+eps_S and original b and S0) " << "with eps_S " << eps_S << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 1 (S+eps_S and original b and S0) " << "with eps_S " << eps_S << endl;

	S = Sa+eps_S;
	mapM_1(coh,out); //result in tempS and cohadj

	//results are stored in the first column of Sb
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,0) = tempS;
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,0) = cohit->getb(); 
		i++;
	}

	/*M1 for S=Sa-eps_S and original number of births in the cohorts and S0=S0a*/
	if(out==1){
		myfile_OC << "step 2 (S-eps_S and original b and S0) " << "with eps_S " << eps_S << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2 (S-eps_S and original b and S0) " << "with eps_S " << eps_S << endl;

	S = Sa-eps_S;
	mapM_1(coh,out);  //result in tempS and cohadj

	//results are stored in the second column of Sb
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,1) = tempS;
	i=1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,1) = cohit->getb(); 
		i++;
	}
		
	/*M1 for S=Sa and adjusted number of births in the cohorts (one by one) and S0=S0a*/
	if(out==1){
		myfile_OC << "step 3 - (2N+2) (S and S0 and b+eps_B (b-eps_B) for each of the cohorts one by one)" << endl;
		myfile_OC << endl;
	}
	S = Sa;
	cohit = coh.begin();
	cohtit = coht.begin();
	int j; //j = number of cohort that is adjusted (j=1,...,coh.size())
	for(j=1;j!=(coh.size()+1);++j){ //do the calculation for every cohort seperately adjusted (at start of loop: coht=coh always)
		eps_B = RCONST(epsilon_b*cohit->getb()); //calculate eps_B
		eps_B_vec(j-1) = eps_B; //store eps_B in eps_B_vec
		if(out==1){
			myfile_OC << "step " << (1+j*2) << " for cohort" << j <<" (S and b+eps_B and S0) " << "with eps_B " << eps_B << endl;
			myfile_OC << endl;
		}
		myfile2 << "step " << (1+j*2) << " for cohort" << j <<" (S and b+eps_B and S0) " << "with eps_B " << eps_B << endl;
		//M1 for adjusted number of births in cohort j (original number + eps_B)
		//coht = coh with only the adjustment for cohort j (at cohtit)
		cohtit->setb(RCONST(cohit->getb()+eps_B));
		mapM_1(coht,out); //result in tempS and cohadj
		myfile2 << "S = " << tempS << endl;
		myfile2 << endl;
		Sb(0,2*j) = tempS; //fill column 2*j+1 (but here start counting from 0 instead of 1 => 2*j)
		i = 1;
		for(cohadjit=cohadj.begin();cohadjit!=cohadj.end();++cohadjit){
			Sb(i,2*j) = cohadjit->getb(); 
			i++;
		}

		//M1 for adjusted number of cohorts in cohort j (original number - eps_B)
		// coht (=coh) with the one adjustment for cohort j (=cohtit)
		if(out==1){
			myfile_OC << "step " << (2+j*2) << " for cohort" << j << " (S and b-eps_B and S0) " << "with eps_B "<< eps_B << endl;
			myfile_OC << endl;
		}
		myfile2 << "step " << (2+j*2) << " for cohort" << j <<" (S and b-eps_B and S0) " << "with eps_B " << eps_B << endl;
		cohtit->setb(RCONST(cohit->getb()-eps_B));
		mapM_1(coht,out);
		myfile2 << "S = " << tempS << endl;
		myfile2 << endl;
		Sb(0,2*j+1) = tempS; //fill column 2*j+2 (but here start counting from 0 instead of 1 => 2*j+1)
		i = 1;
		for(cohadjit=cohadj.begin();cohadjit!=cohadj.end();++cohadjit){
			Sb(i,2*j+1)=cohadjit->getb(); 
			i++;
		}
		cohtit->setb(cohit->getb()); //set number of births back to the original value
		cohit++; //in the next iteration of the loop: use the b-value of the next cohort in coh
		cohtit++; //in the next iteration of the loop: change the next cohort in coht
	}

	//empty the list coht
	cohtit = coht.begin(); 
	while(cohtit!=coht.end())
		cohtit = coht.erase(cohtit);

	/*M1 for S=Sa and original number of births in all the cohorts and S0=S0a+eps_S0*/
	if(out==1){
		myfile_OC << "step 2N+3 (S and b and S0+eps_S0) " << "with eps_S0 " << eps_S0 << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2N+3 (S and b and S0+eps_S0) " << "with eps_S0 " << eps_S0 << endl;

	S0 = RCONST(S0+eps_S0);
	S = Sa;

	mapM_1(coh,out); //result in tempS and cohadj
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,2*coh.size()+2) = tempS; //store the result in the penultimate column
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,2*coh.size()+2) = cohit->getb(); 
		i++;
	}
		
	/*M1 for S=Sa and original number of births in all the cohorts and S0=S0a-eps_S0*/
	if(out==1){
		myfile_OC << "step 2N+4 (S and b and S0-eps_S0) " << "with eps_S0 " << eps_S0 << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2N+4 (S and b and S0-eps_S0) " << "with eps_S0 " << eps_S0 << endl;

	S0 = RCONST(S0a-eps_S0);
	S = Sa;
	mapM_1(coh,out); //result in tempS and cohadj
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,2*coh.size()+3) = tempS; //store the result in the last column
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,2*coh.size()+3) = cohit->getb(); 
		i++;
	}

	myfile2.close();
	S0=S0a;

	//print Sb
	if(out==1){
		myfile_OC << "Sb =" << endl;
		for(int k=0;k!=n1;++k){
			for(int l=0;l!=n3;++l){
				myfile_OC << Sb(k,l) << " ";
			}
			myfile_OC << endl;
		}
		myfile_OC << endl;
	}

	/*calculation of elements of Jacobian*/
	/*elements of column k and row l*/
	int k = 0; //the derivative with respect to S
	for(int l=0;l!=n1;++l){ 
		if(k==l)
			A(l,k) = RCONST(((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_S))-1);
		else
			A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_S));
	}

	k = n2-1; //the derivative with respect to S0
	for(int l=0;l!=n1;++l){
		A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_S0));
	}

	for(k=1;k!=n2-1;++k){ //the derivatives with respect to the b's
		for(int l=0;l!=n1;++l){
			if(k==l)
				A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_B_vec(k-1))-1);
			else
				A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_B_vec(k-1)));
		}
	}

	if(out==1){ //if out=1: print Jacobian A to outputcohorts.txt and to JacGSO.txt
		myfile_OC << "Jac =" << endl;
		for(k=0;k!=n1;++k){
			for(int l=0;l!=n2;++l){
				myfile_OC << A(k,l) << " ";
			}
			myfile_OC << endl;
		}
		myfile_OC << endl;
		
		ofstream myfile;
	    myfile.precision(10);
		myfile.open ("JacGS0.txt", ios::app);
		myfile << "Jac =" <<endl;
		for(k=0;k!=n1;++k){
			for(int l=0;l!=n2;++l){
				myfile << A(k,l) << " ";
			}
			myfile << endl;
		}
		myfile << endl;
		myfile.close();
	}

	return A; //return Jacobian A
}

alglib::real_2d_array JacobianGD(realtype Sa, realtype Da, clist coh, int out){ 
	/* calculates the Jacobian of the map G := mapM_1 - identity (has to be zero for a fixed point)
	   in the point S=Sa, D=Da and list of cohorts = coh (=> number of cells AND the mass-distribution)
	   parameter out (int): = 1 (give full output), = 0 (only give important output)*/

	D = Da;

	alglib::real_2d_array A; //the Jacobian that will be returned
	int n1 = coh.size()+1; //number of rows of the Jacobian
	int n2 = coh.size()+2; //number of columns of the Jacobian
	A.setlength(n1,n2);

	alglib::real_2d_array Sb; //matrix with the results of the calculations for the Jacobian 
	int n3 = 2*coh.size()+4; //number of columns = how many times the map has to be evaluated (1 column = 1 map evaluation)
	Sb.setlength(n1,n3); //number of rows = 1 (S) + number of cohorts (b) 

	ofstream myfile2;
	myfile2.precision(10);
	myfile2.open("mapM1.txt", ios::app); 
	
	realtype eps_D = RCONST(0.01*Da); //eps_D = 1% of D
	realtype eps_S = RCONST(0.01*Sa); //eps_S = 1% of S
	realtype epsilon_b = RCONST(0.01); //epsilon_b = percentage of each b that will be used for eps_B = 1%
	realtype eps_B; //value of epsilon used for calculation of the derivative with respect to a certain b
	                //calculated as epsilon_b (originally 1%) * b for each b  
	                // attention: not the same as eps_b (see constants definitions)
	alglib::real_1d_array eps_B_vec; //vector where all the used eps_B are stored (length = number of cohorts)
	eps_B_vec.setlength(coh.size());
	
	clist coht = coh; // used for the map evalution where one of the b-values of coh is adjusted (for the derivatives with respect to b)
	                // initially coht = coh
	clistit cohit;
	clistit cohadjit;
	clistit cohtit;

	int i; //used for the iteration over the rows of the matrix Sb

	/*M1 for S=Sa+eps_S and original number of births in the cohorts and D=Da*/
	if(out==1){
		myfile_OC << "step 1 (S+eps_S and original b and D) " << "with eps_S " << eps_S << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 1 (S+eps_S and original b and D) " << "with eps_S " << eps_S << endl;
	S = Sa+eps_S;
	mapM_1(coh,out); //result in tempS and cohadj

	//results are stored in the first column of Sb
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,0) = tempS;
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,0) = cohit->getb(); 
		i++;
	}

	/*M1 for S=Sa-eps_S and original number of births in the cohorts and D=Da*/
	if(out==1){
		myfile_OC << "step 2 (S-eps_S and original b and D) " << "with eps_S " << eps_S << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2 (S-eps_S and original b and D) " << "with eps_S " << eps_S << endl;

	S = Sa-eps_S;
	mapM_1(coh,out); //result in tempS and cohadj

	//results are stored in the second column of Sb
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,1) = tempS;
	i=1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,1) = cohit->getb(); 
		i++;
	}
		
	/*M1 for S=Sa and adjusted number of births in the cohorts (one by one) and D=Da*/
	if(out==1){
		myfile_OC << "step 3 - (2N+2) (S and D and b+eps_B (b-eps_B) for each of the cohorts one by one)" << endl;
		myfile_OC << endl;
	}
	S = Sa;
	cohit = coh.begin();
	cohtit = coht.begin();
	int j; //j = number of cohort that is adjusted (j=1,...,coh.size())
	for(j=1;j!=(coh.size()+1);++j){ //do the calculation for every cohort seperately adjusted (at start of loop: coht=coh always)
		eps_B = RCONST(epsilon_b*cohit->getb()); //calculate eps_B
		eps_B_vec(j-1) = eps_B; //store eps_B in eps_B_vec
		if(out==1){
			myfile_OC << "step " << (1+j*2) << " for cohort" << j <<" (S and b+eps_B and D) " << "with eps_B " << eps_B << endl;
			myfile_OC << endl;
		}
		myfile2 << "step " << (1+j*2) << " for cohort" << j <<" (S and b+eps_B and D) " << "with eps_B " << eps_B << endl;
		//M1 for adjusted number of births in cohort j (original number + eps_B)
		//coht = coh with only the adjustment for cohort j (at cohtit)
		cohtit->setb(RCONST(cohit->getb()+eps_B));
		mapM_1(coht,out); //result in tempS and cohadj
		myfile2 << "S = " << tempS << endl;
		myfile2 << endl;

		Sb(0,2*j) = tempS; //fill column 2*j+1 (but here start counting from 0 instead of 1 => 2*j)
		i = 1;
		for(cohadjit=cohadj.begin();cohadjit!=cohadj.end();++cohadjit){
			Sb(i,2*j) = cohadjit->getb(); 
			i++;
		}

		//M1 for adjusted number of cohorts in cohort j (original number - eps_B)
		// coht (=coh) with the one adjustment for cohort j (=cohtit)
		if(out==1){
			myfile_OC << "step " << (2+j*2) << " for cohort" << j << " (S and b-eps_B and D) " << "with eps_B "<< eps_B << endl;
			myfile_OC << endl;
		}
		myfile2 << "step " << (2+j*2) << " for cohort" << j <<" (S and b-eps_B and D) " << "with eps_B " << eps_B << endl;
		cohtit->setb(RCONST(cohit->getb()-eps_B));
		mapM_1(coht,out);

		myfile2 << "S = " << tempS << endl;
		myfile2 << endl;

		Sb(0,2*j+1) = tempS; //fill column 2*j+2 (but here start counting from 0 instead of 1 => 2*j+1)
		i = 1;
		for(cohadjit=cohadj.begin();cohadjit!=cohadj.end();++cohadjit){
			Sb(i,2*j+1)=cohadjit->getb(); 
			i++;
		}

		cohtit->setb(cohit->getb()); //set number of births back to the original value
		cohit++; //in the next iteration of the loop: use the b-value of the next cohort in coh
		cohtit++; //in the next iteration of the loop: change the next cohort in coht
	}

	//empty the list coht
	cohtit = coht.begin(); 
	while(cohtit!=coht.end())
		cohtit = coht.erase(cohtit);

	/*M1 for S=Sa and original number of births in all the cohorts and D=Da+eps_D*/
	if(out==1){
		myfile_OC << "step 2N+3 (S and b and D+eps_D) " << "with eps_D " << eps_D << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2N+3 (S and b and D+eps_D) " << "with eps_D " << eps_D << endl;

	D = RCONST(D+eps_D);
	S = Sa;
	mapM_1(coh,out); //result in tempS and cohadj
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,2*coh.size()+2) = tempS; //store the result in the penultimate column
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,2*coh.size()+2) = cohit->getb(); 
		i++;
	}
		
	/*M1 for S=Sa and original number of births in all the cohorts and D=Da-eps_D*/
	if(out==1){
		myfile_OC << "step 2N+4 (S and b and D-eps_D) " << "with eps_D " << eps_D << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2N+4 (S and b and D-eps_D) " << "with eps_D0 " << eps_D << endl;

	D = RCONST(Da-eps_D);
	S = Sa;
	mapM_1(coh,out); //result in tempS and cohadj
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,2*coh.size()+3) = tempS; //store the result in the last column
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,2*coh.size()+3) = cohit->getb(); 
		i++;
	}

	myfile2.close();

	D = Da;

	//print Sb
	if(out==1){
		myfile_OC << "Sb =" << endl;
		for(int k=0;k!=n1;++k){
			for(int l=0;l!=n3;++l){
				myfile_OC << Sb(k,l) << " ";
			}
			myfile_OC << endl;
		}
		myfile_OC << endl;
	}


	/*calculation of elements of Jacobian*/
	/*elements of column k and row l*/
	int k = 0; //the derivative with respect to S
	for(int l=0;l!=n1;++l){ 
		if(k==l)
			A(l,k) = RCONST(((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_S))-1);
		else
			A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_S));
	}

	k = n2-1; //the derivative with respect to D
	for(int l=0;l!=n1;++l){
		A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_D));
	}

	for(k=1;k!=n2-1;++k){ //the derivatives with respect to the b's
		for(int l=0;l!=n1;++l){
			if(k==l)
				A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_B_vec(k-1))-1);
			else
				A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_B_vec(k-1)));
		}
	}

	if(out==1){ //if out=1: print Jacobian A to outputcohorts.txt and to JacGSO.txt
		myfile_OC << "Jac =" << endl;
		for(k=0;k!=n1;++k){
			for(int l=0;l!=n2;++l){
				myfile_OC << A(k,l) << " ";
			}
			myfile_OC << endl;
		}
		myfile_OC << endl;
		
		ofstream myfile;
	    myfile.precision(10);
		myfile.open ("JacGD.txt", ios::app);
		myfile << "Jac =" <<endl;
		for(k=0;k!=n1;++k){
			for(int l=0;l!=n2;++l){
				myfile << A(k,l) << " ";
			}
			myfile << endl;
		}
		myfile << endl;
		myfile.close();
	}

	return A; //return Jacobian A
}

alglib::real_2d_array JacobianM1(realtype Sa, clist coh, int out){ 
	/* calculates the Jacobian of mapM_1 
	   in the point S=Sa and list of cohorts = coh (=> number of cells AND internal states)
	   parameter out (int): = 1 (give full output), = 0 (only give important output)*/

	alglib::real_2d_array A; //the Jacobian that will be returned
	int n1 = coh.size()+1; //number of rows of the Jacobian
	int n2 = coh.size()+1; //number of columns of the Jacobian
	A.setlength(n1,n2);

	alglib::real_2d_array Sb; //matrix with the results of the calculations for the Jacobian 
	int n3 = 2*coh.size()+2; //number of columns = how many times the map has to be evaluated (1 column = 1 map evaluation)
	Sb.setlength(n1,n3); //number of rows = 1 (S) + number of cohorts (b) 

	ofstream myfile2;
	myfile2.precision(10);
	myfile2.open("mapM1.txt", ios::app); 
	
	realtype eps_S = RCONST(0.01*Sa); //eps_S = 1% of S
	realtype epsilon_b = RCONST(0.01); //epsilon_b = percentage of each b that will be used for eps_B = 1%
	realtype eps_B; //value of epsilon used for calculation of the derivative with respect to a certain b
	                //calculated as epsilon_b (originally 1%) * b for each b  
	                // attention: not the same as eps_b (see constants definitions)
	alglib::real_1d_array eps_B_vec; //vector where all the used eps_B are stored (length = number of cohorts)
	eps_B_vec.setlength(coh.size());
	
	clist coht = coh; // used for the map evalution where one of the b-values of coh is adjusted (for the derivatives with respect to b)
	                // initially coht = coh
	clistit cohit;
	clistit cohadjit;
	clistit cohtit;

	int i; //used for the iteration over the rows of the matrix Sb

	/*M1 for S=Sa+eps_S and original number of births in the cohorts*/
	if(out==1){
		myfile_OC << "step 1 (S+eps_S and original b) " << "with eps_S " << eps_S << endl;
		myfile_OC << endl;
	}

	myfile2 << "step 1 (S+eps_S and original b) " << "with eps_S " << eps_S << endl;

	S = Sa+eps_S;
	mapM_1(coh,out); //result in tempS and cohadj

	//results are stored in the first column of Sb
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,0) = tempS;
	i = 1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,0) = cohit->getb(); 
		i++;
	}

	/*M1 for S=Sa-eps_S and original number of births in the cohorts*/
	if(out==1){
		myfile_OC << "step 2 (S-eps_S and original b) " << "with eps_S " << eps_S << endl;
		myfile_OC << endl;
	}
	myfile2 << "step 2 (S-eps_S and original b) " << "with eps_S " << eps_S << endl;

	S = Sa-eps_S;
	mapM_1(coh,out);  //result in tempS and cohadj

	//results are stored in the second column of Sb
	myfile2 << "S = " << tempS << endl;
	myfile2 << endl;
	Sb(0,1) = tempS;
	i=1;
	for(cohit=cohadj.begin();cohit!=cohadj.end();++cohit){
		Sb(i,1) = cohit->getb(); 
		i++;
	}
		
	/*M1 for S=Sa and adjusted number of births in the cohorts (one by one)*/
	if(out==1){
		myfile_OC << "step 3 - (2N+2) (S and b+eps_B (b-eps_B) for each of the cohorts one by one)" << endl;
		myfile_OC << endl;
	}
	S = Sa;
	cohit = coh.begin();
	cohtit = coht.begin();
	int j; //j = number of cohort that is adjusted (j=1,...,coh.size())
	for(j=1;j!=(coh.size()+1);++j){ //do the calculation for every cohort seperately adjusted (at start of loop: coht=coh always)
		eps_B = RCONST(epsilon_b*cohit->getb()); //calculate eps_B
		eps_B_vec(j-1) = eps_B; //store eps_B in eps_B_vec
		if(out==1){
			myfile_OC << "step " << (1+j*2) << " for cohort" << j <<" (S and b+eps_B) " << "with eps_B " << eps_B << endl;
			myfile_OC << endl;
		}
		myfile2 << "step " << (1+j*2) << " for cohort" << j <<" (S and b+eps_B) " << "with eps_B " << eps_B << endl;
		//M1 for adjusted number of births in cohort j (original number + eps_B)
		// coht = coh with only the adjustment for cohort j (at cohtit)
		cohtit->setb(RCONST(cohit->getb()+eps_B));

		mapM_1(coht,out); //result in tempS and cohadj
		myfile2 << "S = " << tempS << endl;
		myfile2 << endl;

		Sb(0,2*j) = tempS; //fill column 2*j+1 (but here start counting from 0 instead of 1 => 2*j)
		i = 1;
		for(cohadjit=cohadj.begin();cohadjit!=cohadj.end();++cohadjit){
			Sb(i,2*j) = cohadjit->getb(); 
			i++;
		}

		//M1 for adjusted number of cohorts in cohort j (original number - eps_B)
		// coht (=coh) with the one adjustment for cohort j (=cohtit)
		if(out==1){
			myfile_OC << "step " << (2+j*2) << " for cohort" << j << " (S and b-eps_B) " << "with eps_B "<< eps_B << endl;
			myfile_OC << endl;
		}
		myfile2 << "step " << (2+j*2) << " for cohort" << j <<" (S and b-eps_B) " << "with eps_B " << eps_B << endl;
		cohtit->setb(RCONST(cohit->getb()-eps_B));
		mapM_1(coht,out);

		myfile2 << "S = " << tempS << endl;
		myfile2 << endl;

		Sb(0,2*j+1) = tempS; //fill column 2*j+2 (but here start counting from 0 instead of 1 => 2*j+1)
		i = 1;
		for(cohadjit=cohadj.begin();cohadjit!=cohadj.end();++cohadjit){
			Sb(i,2*j+1)=cohadjit->getb(); 
			i++;
		}

		cohtit->setb(cohit->getb()); //set number of births back to the original value
		cohit++; //in the next iteration of the loop: use the b-value of the next cohort in coh
		cohtit++; //in the next iteration of the loop: change the next cohort in coht
	}

	//empty the list coht
	cohtit = coht.begin(); 
	while(cohtit!=coht.end())
		cohtit = coht.erase(cohtit);

	myfile2.close();

	//print Sb
	if(out==1){
		myfile_OC << "Sb =" << endl;
		for(int k=0;k!=n1;++k){
			for(int l=0;l!=n3;++l){
				myfile_OC << Sb(k,l) << " ";
			}
			myfile_OC << endl;
		}
		myfile_OC << endl;
	}


	/*calculation of elements of Jacobian*/
	/*elements of column k and row l*/
	int k = 0; //the derivative with respect to S
	for(int l=0;l!=n1;++l)
		A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_S));

	for(k=1;k!=n2-1;++k){ //the derivatives with respect to the b's
		for(int l=0;l!=n1;++l)
			A(l,k) = RCONST((Sb(l,2*k)-Sb(l,2*k+1))/(2*eps_B_vec(k-1)));
	}

	if(out==1){ //if out=1: print Jacobian A to outputcohorts.txt and to JacGSO.txt
		myfile_OC << "Jac=[";
		for(k=0;k!=n1;++k){
			for(int l=0;l!=n2;++l){
				myfile_OC << " " << A(k,l);
			}
			myfile_OC << ";" << endl;
		}
		myfile_OC << "];" << endl;
		
		ofstream myfile;
	    myfile.precision(10);
		myfile.open ("JacGS0.txt", ios::app);
		myfile << "Jac=[";
		for(k=0;k!=n1;++k){
			for(int l=0;l!=n2;++l){
				myfile << " " << A(k,l);
			}
			myfile << ";" << endl;
		}
		myfile << "];" << endl;
		myfile.close();
	}

	return A; //return Jacobian A
}

int contfpS0(realtype &Sa, realtype &S0a, clist &coh, realtype step_start, realtype step_m, realtype step_M, int Newtonmax, realtype FunTolerance,realtype VarTolerance,int dir, int a, int out){ 
	/*start values of fixed point (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + S0)
	 parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-001)
	 parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-005)
	 parameter step_M: maximum value of the stepsize of the continuation (eg 1.0)
	 parameter Newtonmax (int): maximum number of Newton steps (eg 10)
	 parameter FunTolerance: precision to be satisfied in Newtoncorrection to have convergence (eg 1.0e-006) for L1-norm of R
	 parameter VarTolerance: precision to be satisfied in Newtoncorrection to have convergence (eg 1.0e-006) for L1-norm of cor
	 parameter dir: gives the direction of the continuation at start -> increasing S0 if = 1, decreasing S0 if = -1 
	 parameter a (int): number of fixed points to calculate 
	 parameter out (int): = 1 (yes, give full output), = 0 (no, give only important output about the continuation)
	 returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	 changes to original values of Sa, S0a and coh made in function!*/

	int j; //number of Newton steps taken
	int p = 0; //number of fixed points calculated
	int q = 0; //used to determine if vector v has to be calculated (=0: yes, =1: no)
	realtype step = step_start; //stepsize continuation
	realtype btot = RCONST(0);

	clistit cohit = coh.begin();
	ofstream myfile;
	ofstream myfile1;
	ofstream myfile2;
	myfile.open ("cont.txt", ios::app); 
	myfile.precision(10);
	myfile1.open("continuation.txt", ios::app);
	myfile1.precision(10);
	myfile2.open("cont_S0NSb.txt", ios::app);
	myfile2.precision(10);
	myfile << p << " " << Sa << " " << S0a << " " << endl;
	myfile1<< p << " " << Sa << " " << S0a << " " << endl;
	myfile2<< S0a << " " << coh.size() << " " << Sa << " ";
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		btot += cohit->getb();
	}
	myfile << endl;
	myfile1 << endl;
	myfile2 << btot << endl;
	btot = 0;

	if((dir!=1) && (dir!=-1)){
		myfile << "wrong value for the parameter dir: give 1 for increasing S0 and -1 for decreasing S0 in the continuation" << endl;
		myfile.close();
		return 1;
	}
	if((out!=1) && (out!=0)){
		myfile << "wrong value for the parameter out: give 1 for full output and 0 for only the important output" << endl;
		myfile.close();
		return 1;
	}	

	clist cohp;
	clistit cohpit;
	cohort b;
	realtype Sp, S0p;
	alglib::real_2d_array JG;
	alglib::real_2d_array JGa;
	alglib::real_1d_array rl;
	alglib::real_1d_array v;
	alglib::real_1d_array vn;
	alglib::real_1d_array vnoud;
	alglib::real_1d_array bp;
	alglib::real_1d_array vp;
	alglib::real_1d_array moud;
	alglib::real_1d_array mnew;
	alglib::ae_int_t info;
	alglib::densesolverreport rep;
	alglib::real_2d_array J1;
	alglib::real_2d_array J2;
	alglib::real_1d_array R; 
	alglib::real_1d_array cor;
	int iadj = 0; //=1 if stepsize for prediction has to be changed 
	realtype normR;
	realtype normcor;

	
	//FIRST CONTINUATION STEP
	
	//store mass values of cohorts of current fixed point (coh) in mnew
	mnew.setlength(coh.size());
	int it = 0;
	for(cohit=coh.begin();cohit!=coh.end();cohit++){
		mnew(it) = cohit->getMass();
		it++;
	}

	myfile << "calculate Jacobian" << endl;
	myfile << "Sa = " << Sa << ", S0a = " << S0a << endl;
	JG = JacobianGS0(Sa,S0a,coh,out);
	myfile << " Jacobian:" <<endl;
	for(int k=0;k!=JG.rows();++k){
		for(int l=0;l!=JG.cols();++l)
			myfile << JG(k,l) << " ";
		myfile << endl;
	}
	myfile << endl;

	JGa.setlength(JG.cols(),JG.cols());
	rl.setlength(JG.cols());
	for(int k=0;k<JG.rows();++k){
		for(int l=0;l<JG.cols();++l)
			JGa(k,l) = JG(k,l);
		rl(k) = 0;
	}
	rl(JG.rows()) = 1;
	for(int l=0;l<JG.cols();++l)
		JGa(JG.rows(),l) = 1;
	alglib::rmatrixsolve(JGa,JG.cols(),rl,info,rep,v);
	myfile << " v (unstandardised)=" << endl;
	for(int k=0;k<JG.cols();++k)
		myfile << v(k) << " ";
	myfile << endl;
	vn = normalize(v);
	myfile << "v (normalized)=" << endl;
	for(int k=0;k<vn.length();++k)
		myfile << vn(k) << " ";
	myfile << endl;

	if(((vn(vn.length()-1)>0)&&(dir!=1))||((vn(vn.length()-1)<0)&&(dir!=-1))){
		for(int k=0;k<vn.length();++k)
			vn(k) *= -1;
		myfile << "adjust direction of v: yes" << endl;
	}
	else{
		myfile << "adjust direction of v: no" <<endl;
	}

	// calculation of predicted values:
	Sp = RCONST(Sa+step*vn(0));
	S0p = RCONST(S0a+step*vn(vn.length()-1));
	myfile << "predicted values:" << endl;
	myfile << Sp << " " << S0p;
	int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
	bp.setlength(coh.size());
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
		b.set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
		cohp.push_back(b);
		myfile << " " << bp(nc-1);
		nc++;
	}
	myfile<<endl;
		
	//adjust stepsize (stepsize/2) because one of the predicted values is <=0? if one of the predicted values is <=0: set (int) iadj=1
	if((Sp<=0)||(S0p<=0))
		iadj = 1;
	else{
		for(int k=0;((k<bp.length())&&(iadj!=1));++k){
			if(bp(k)<=0)
				iadj = 1;
		}
	}
		
	while(iadj==1){
		myfile << "one of the predicted values is <=0, this is not allowed, so adjust stepsize (stepsize/2)" <<endl;
		if(step==step_m){
			myfile << "stepsize already has reached minimum value " << step_m << endl;
			myfile << "end continuation" << endl;
			return 1;
		}
		step = RCONST(step/2.0);
		if(step<step_m){
			step = step_m;
			myfile << "minimal step reached" << endl;
		}
		myfile << "new stepsize = " << step << endl;
			
		Sp = RCONST(Sa+step*vn(0));
		S0p = RCONST(S0a+step*vn(vn.length()-1));
		myfile << "predicted values:" << endl;
		myfile << Sp << " " << S0p;
		cohpit = cohp.begin();
		int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
		bp.setlength(coh.size()); //replace bp
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
			cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
			myfile << " " << cohpit->getb();
			cohpit++;
			nc++;
		}
		myfile << endl;
		iadj = 0;
		if((Sp<=0)||(S0p<=0)){
			iadj = 1;
		}
		else{
			for(int k=0;((k<bp.length())&&(iadj!=1));++k){
				if(bp(k)<=0)
					iadj = 1;
			}
		}
	}

	//adjust mesh points (change of cohp and Sp)
	myfile << "ADJUST MESH OF PREDICTION" << endl;
	S = RCONST(Sp);
	S0 = RCONST(S0p);
	myfile << "Sa = " << Sa << ", S0a = " << S0a << endl;
	integrate_allcohorts(cohp,out,0);
	myfile << endl;
	if(tempS<=0)
		Sp = 0.000001;
	else
		Sp = tempS;
	cohp = cohnew;
	myfile << "adjusted Sp: " << Sp << endl;
	myfile << "adjusted number of cohorts: " << cohp.size() << endl;
	myfile << "adjusted mesh points:" << endl;
	for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
		myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
	}
	myfile << endl;

	//Newton-correction 

	// first step
	j = 1;

	//calculation of Jacobian J1
	J1 = JacobianGS0(Sp,S0p,cohp,out);
	myfile << " Jacobian J1:" << endl;
	for(int k=0;k!=J1.rows();++k){
		for(int l=0;l!=J1.cols();++l)
			myfile << J1(k,l) << " ";
		myfile << endl;
	}
	myfile << endl;

	//adjust J1 to J2 to calculate vp
	J2.setlength(J1.cols(),J1.cols());
	rl.setlength(J1.cols());
	for(int k=0;k<J1.rows();++k){
		for(int l=0;l<J1.cols();++l)
			J2(k,l) = J1(k,l);
		rl(k) = 0;
	}
	rl(J1.rows()) = 1;
	for(int l=0;l<J1.cols();++l)
		J2(J1.rows(),l) = 1;
	alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
	myfile << " vp (unstandardised)=" << endl;
	for(int k=0;k<J1.cols();++k)
		myfile << v(k) << " ";
	myfile << endl;
	vp = normalize(v);
	myfile << "vp (normalized)=" << endl;
	for(int k=0;k<vp.length();++k)
		myfile << vp(k) << " ";
	myfile << endl;

	//adjust J2 to (J1 vp)^T
	for(int l=0;l<J1.cols();++l)
		J2(J1.rows(),l) = vp(l);

	//calculation of right-hand side R = predicted values - values after 1 map iteration 
	S = Sp;
	S0 = S0p;
	mapM_1(cohp,out);
	R.setlength(J2.cols());
	if(tempS<=0){
		myfile << "values after 1 map iteration (S,b): " << 0.000001;
		R(0) = RCONST(Sp-0.000001);
	}
	else{
		myfile << "values after 1 map iteration (S,b): " << tempS;
		R(0) = RCONST(Sp-tempS);
	}

	myfile << endl;

	it = 1;
	cohpit = cohp.begin();
	for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
		myfile << " " << cohit->getb(); 
		R(it) = RCONST(cohpit->getb()-cohit->getb());
		it++;
		cohpit++;
	}
	myfile << endl;
	R(J2.cols()-1) = RCONST(0);
	myfile << "values right-hand side R =";
	normR = RCONST(0);
	for(int l=0;l<R.length();++l){
		myfile << " " << R(l);
		normR += abs(R(l));
	}
	myfile << endl;

	//solve equation J2*cor=R => cor
	alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
	myfile << "values correction = ";
	normcor = RCONST(0);
	for(int l=0;l<cor.length();++l){
		myfile << " " <<cor(l);
		normcor += abs(cor(l));
	}
	myfile << endl;
	myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
	myfile << "number of Newton steps taken = " << j << endl;

	// next possible steps
	while( (j<=Newtonmax) && ((normcor>VarTolerance)||(normR>FunTolerance)) ){
		//check if R & cor are small enough (if not: the Newton correction step has to be repeated) 
		//(if number of Newton steps taken <= Newtonmax, if more steps are already taken: adjust stepsize and retake prediction)
		myfile << "Newton correction step has to be repeated " << endl;
		// adjust prediction
		Sp += cor(0);
		S0p += cor(cor.length()-1);
		myfile << "adjusted prediction = " << Sp;
		it = 1;
		for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
			cohpit->setb(RCONST(cohpit->getb()+cor(it)));
			it++;
			myfile << " " << cohpit->getb();
		}
		myfile << " " << S0p << endl;
		myfile << endl;

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		S0 = S0p;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp: " << Sp << endl;
		myfile << "adjusted number of cohorts: " << cohp.size() << endl;
		myfile << "adjusted mesh points:" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit)
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		myfile << endl;

		//Newton-correction 

		//calculation of Jacobian J2
		J1 = JacobianGS0(Sp,S0p,cohp,out);
		myfile << " Jacobian J1:" << endl;
		for(int k=0;k!=J1.rows();++k){
			for(int l=0;l!=J1.cols();++l)
				myfile << J1(k,l) << " ";
			myfile << endl;
		}
		myfile << endl;

		//adjust J2 to calculate vp
		J2.setlength(J1.cols(),J1.cols());
		rl.setlength(J1.cols());
		for(int k=0;k<J1.rows();++k){
			for(int l=0;l<J1.cols();++l)
				J2(k,l) = J1(k,l);
			rl(k) = 0;
		}
		rl(J1.rows()) = 1;
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = 1;
		alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
		myfile << " vp (unstandardised)=" << endl;
		for(int k=0;k<J1.cols();++k)
			myfile << v(k) << " ";
		myfile << endl;
		vp = normalize(v);
		myfile << "vp (normalized)=" << endl;
		for(int k=0;k<vp.length();++k)
			myfile << vp(k) << " ";
		myfile << endl;

		//adjust J2 to (J1 vp)^T
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = vp(l);

		//calculation of right-hand side R = predicted values - values after 1 map iteration 
		S = Sp;
		S0 = S0p;
		mapM_1(cohp,out);
		R.setlength(J2.cols());
		if(tempS<=0){
			myfile << "values after 1 map iteration (S,b): " << 0.000001;
			R(0) = RCONST(Sp-0.000001);
		}
		else{
			myfile << "values after 1 map iteration (S,b): " << tempS;
			R(0) = RCONST(Sp-tempS);
		}

		it = 1;
		cohpit = cohp.begin();
		for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
			myfile << " " << cohit->getb(); 
			R(it) = RCONST(cohpit->getb()-cohit->getb());
			it++;
			cohpit++;
		}
		myfile << endl;
		R(J2.cols()-1) = RCONST(0);
		myfile << "values right-hand side R =";
		normR = RCONST(0);
		for(int l=0;l<R.length();++l){
			myfile << " " << R(l);
			normR += abs(R(l));
		}
		myfile << endl;

		//solve equation J2*cor=R => cor
		alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
		myfile << "values correction = ";
		normcor = RCONST(0);
		for(int l=0;l<cor.length();++l){
			myfile << " " << cor(l);
			normcor += abs(cor(l));
		}
		myfile << endl;
		j++;
		myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
		myfile << "number of Newton steps taken = " << j << endl;
		myfile << endl;
	}
		
	if((normcor<=VarTolerance)&&(normR<=FunTolerance)){
		//IF SMALL ENOUGH: new fixed point = predicted + correction (with adjusted mesh points)
		myfile << "correction small enough " << endl;

		// adjust prediction:
		Sp += cor(0);
		S0p += cor(cor.length()-1);
		myfile << "adjusted prediction = " << Sp;
		int it = 1;
		for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
			cohpit->setb(RCONST(cohpit->getb()+cor(it)));
			it++;
			myfile << " " << cohpit->getb();
		}
		myfile << " " << S0p << endl;
		myfile << endl;

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		S0 = S0p;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp (of new fixed point): " << Sp << endl;
		myfile << "adjusted number of cohorts (of new fixed point): " << cohp.size() << endl;
		myfile << "adjusted mesh points (of new fixed point):" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		}
		myfile << endl;

		Sa = Sp;
		S0a = S0p;
		coh.assign(cohp.begin(),cohp.end());
		p++; //number of calculated fixed points + 1
		myfile << "RESULTS continuation: " << endl;
		myfile << p << " " << Sa << " " << S0a << " " << endl;
		myfile1 << p << " " << Sa << " " << S0a << " " << endl;
		myfile2 << S0a << " " << coh.size() << " " << Sa << " "; 
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			btot += cohit->getb();
		}
		myfile << endl;
		myfile1 << endl;
		myfile2 << btot << endl;
		btot = 0;

		//adjust stepsize according to number of Newton steps taken
		if(j>=4){
			step = RCONST(step/2.0); 
			myfile << "more than 4 or 4 Newton steps needed, so stepsize divided by 2" << endl;
			if(step<step_m){
				step = step_m;
				myfile << "minimal Newton step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			myfile << endl;
			myfile << endl;
		}
		else{
			step = RCONST(step*1.3);
			myfile << "less than 4 Newton steps needed, so stepsize * 1.3" << endl;
			if(step>step_M){
				step = step_M;
				myfile << "maximal Newton step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			myfile <<endl;
			myfile <<endl;
		}
	}

	else{ //more than Newtonmax steps taken, so adjust stepsize and retake prediction
		if(step==step_m){
			myfile << "more than " << Newtonmax << " Newton steps needed and stepsize has reached minimum value " << step_m << endl;
			myfile << "end continuation" << endl;
			return 1;
		}
		step = RCONST(step/2.0); 
		if(step<step_m){
			step = step_m;
			myfile << "minimal Newton step reached" << endl;
		}
		myfile << "more than " << Newtonmax << " Newton steps needed, so stepsize/2 for prediction and tried again" << endl;
		myfile << endl;
		q = 1;
	}


	// LOOP OF NEXT CONTINUATION STEPS (difference with first one: dot product must be positive with tangent vector of previous step to determine direction of new tangent vector for prediction)
	while(p<a){ //loop of calculation of fixed points
		myfile << "q = " << q << endl;
		if(q==0){
			//store mass values of cohorts of previous fixed point in moud (mnew=>moud)
			moud.setlength(mnew.length());
			for(int k=0;k!=mnew.length();++k)
				moud(k) = mnew(k);
			//store mass values of cohorts of current fixed point (coh) in mnew
			mnew.setlength(coh.size());
			it = 0;
			for(cohit=coh.begin();cohit!=coh.end();cohit++){
				mnew(it) = cohit->getMass();
				it++;
			}
			//store tangent vector at previous fixed point (vn) in vnoud
			vnoud.setlength(vn.length());
			for(int k=0;k!=vn.length();++k)
				vnoud(k) = vn(k);

			JG = JacobianGS0(Sa,S0a,coh,out);
			myfile << " Jacobian:" << endl;
			for(int k=0;k!=JG.rows();++k){
				for(int l=0;l!=JG.cols();++l)
					myfile << JG(k,l) << " ";
				myfile << endl;
			}
			myfile << endl;

			JGa.setlength(JG.cols(),JG.cols());
			rl.setlength(JG.cols());
			for(int k=0;k<JG.rows();++k){
				for(int l=0;l<JG.cols();++l)
					JGa(k,l) = JG(k,l);
				rl(k) = 0;
			}
			rl(JG.rows()) = 1;
			for(int l=0;l<JG.cols();++l)
				JGa(JG.rows(),l) = 1;
			alglib::rmatrixsolve(JGa,JG.cols(),rl,info,rep,v);
			myfile << " v (unstandardised)=" << endl;
			for(int k=0;k<JG.cols();++k)
				myfile << v(k) << " ";
			myfile << endl;
			vn = normalize(v);
			myfile << "v (normalized)=" << endl;
			for(int k=0;k<vn.length();++k)
				myfile << vn(k) << " ";
			myfile << endl;

			//check direction of vn: dot product with vnoud must be positive, otherwise: adjust direction
			if(signdotproduct(vnoud,vn,moud,mnew,out)==0){
				myfile << "problem with sizes of vectors for signdotproduct" << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			else{
				if(signdotproduct(vnoud,vn,moud,mnew,out)==+1)
					myfile << "adjust direction of v: no" << endl;
				else{
					if(signdotproduct(vnoud,vn,moud,mnew,out)==-1){
						myfile << "adjust direction of v: yes" << endl;
						for(int k=0;k<vn.length();++k)
							vn(k) *= -1;
					}
					else{
						myfile << "problem with signdotproduct" << endl;
						myfile << "end continuation" << endl;
						return 1;
					}
				}
			}
		}
		q = 0;

		// calculation of predicted values:
		myfile << "step = " << step << ", Sa = " << Sa << ", S0a = " << S0a << endl;
		myfile << "vn = " << endl;
		for(int k=0; k<vn.length(); ++k)
			myfile << vn(k) << " ";
		myfile << endl;
		myfile << "coh = " << endl;
		for(cohit=coh.begin();cohit!=coh.end();++cohit){ 
				myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			}
		myfile << endl;

		Sp = RCONST(Sa+step*vn(0));
		S0p = RCONST(S0a+step*vn(vn.length()-1));
		myfile << "predicted values:" << endl;
		myfile << Sp << " " << S0p;
		cohp.assign(coh.begin(),coh.end());
		cohpit = cohp.begin();
		int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
		bp.setlength(coh.size());
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
			cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
			myfile << " " << cohpit->getb();
			cohpit++;
			nc++;
		}
		myfile << endl;

		//adjust stepsize (stepsize/2) because one of the predicted values is <=0? if one of the predicted values is <=0: set (int) iadj=1
		iadj = 0;
		if((Sp<=0)||(S0p<=0))
			iadj = 1;
		else{
			for(int k=0;((k<bp.length())&&(iadj!=1));++k){
				if(bp(k)<=0)
					iadj = 1;
			}
		}
		
		while(iadj==1){
			myfile << "one of the predicted values is <=0, this is not allowed, so adjust stepsize (stepsize/2)" << endl;
			if(step==step_m){
				myfile << "stepsize already has reached minimum value " << step_m << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			step = RCONST(step/2.0);
			if(step<step_m){
				step = step_m;
				myfile << "minimal step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			
			Sp = RCONST(Sa+step*vn(0));
			S0p = RCONST(S0a+step*vn(vn.length()-1));
			myfile << "predicted values:" << endl;
			myfile << Sp << " " << S0p;
			cohp.assign(coh.begin(),coh.end());
			cohpit = cohp.begin();
			int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
			bp.setlength(coh.size());
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
				cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
				myfile << " " << cohpit->getb();
				cohpit++;
				nc++;
			}
			myfile << endl;
			iadj = 0;
			if((Sp<=0)||(S0p<=0))
				iadj = 1;
			else{
				for(int k=0;((k<bp.length())&&(iadj!=1));++k){
					if(bp(k)<=0)
						iadj = 1;
				}
			}
		}

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		S0 = S0p;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp: " << Sp << endl;
		myfile << "adjusted number of cohorts: " << cohp.size() << endl;
		myfile << "adjusted mesh points:" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit)
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		myfile << endl;

		//Newton-correction 

		// first step
		j = 1;

		//calculation of Jacobian J2
		J1 = JacobianGS0(Sp,S0p,cohp,out);
		myfile << " Jacobian J1:" << endl;
		for(int k=0;k!=J1.rows();++k){
			for(int l=0;l!=J1.cols();++l)
				myfile << J1(k,l) << " ";
			myfile << endl;
		}
		myfile << endl;

		//adjust J2 to calculate vp
		J2.setlength(J1.cols(),J1.cols());
		rl.setlength(J1.cols());
		for(int k=0;k<J1.rows();++k){
			for(int l=0;l<J1.cols();++l)
				J2(k,l) = J1(k,l);
			rl(k) = 0;
		}
		rl(J1.rows()) = 1;
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = 1;
		alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
		myfile << " vp (unstandardised)=" << endl;
		for(int k=0;k<J1.cols();++k)
			myfile << v(k) << " ";
		myfile << endl;
		vp = normalize(v);
		myfile << "vp (normalized)=" << endl;
		for(int k=0;k<vp.length();++k)
			myfile << vp(k) << " ";
		myfile << endl;

		//adjust J2 to (J1 vp)^T
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = vp(l);

		//calculation of right-hand side R = predicted values - values after 1 map iteration 
		S = Sp;
		S0 = S0p;
		mapM_1(cohp,out);
		R.setlength(J2.cols());
		if(tempS<=0){
			myfile << "values after 1 map iteration (S,b): " << 0.000001;
			R(0) = RCONST(Sp-0.000001);
		}
		else{
			myfile << "values after 1 map iteration (S,b): " << tempS;
			R(0) = RCONST(Sp-tempS);
		}

		int it = 1;
		cohpit = cohp.begin();
		for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
			myfile << " " << cohit->getb(); 
			R(it) = RCONST(cohpit->getb()-cohit->getb());
			it++;
			cohpit++;
		}
		myfile << endl;
		R(J2.cols()-1) = RCONST(0);
		myfile << "values right-hand side R =";
		normR = RCONST(0);
		for(int l=0;l<R.length();++l){
			myfile << " " << R(l);
			normR += abs(R(l));
		}
		myfile << endl;

		//solve equation J2*cor=R => cor
		alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
		myfile << "values correction = ";
		normcor = RCONST(0);
		for(int l=0;l<cor.length();++l){
			myfile << " " << cor(l);
			normcor += abs(cor(l));
		}
		myfile << endl;
		myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
		myfile << "number of Newton steps taken = " << j << endl;

		// next possible steps
		while((j<=Newtonmax)&&((normcor>VarTolerance)||(normR>FunTolerance))){
			//check if R & cor are small enough (if not: the Newton correction step has to be repeated) 
			//(if number of Newton steps taken <= Newtonmax, if more steps are already taken: adjust stepsize and retake prediction)
			myfile << "Newton correction step has to be repeated " << endl;
			//adjust prediction:
			Sp += cor(0);
			S0p += cor(cor.length()-1);
			myfile << "adjusted prediction = " << Sp;
			it = 1;
			for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
				cohpit->setb(RCONST(cohpit->getb()+cor(it)));
				it++;
				myfile << " " << cohpit->getb();
			}
			myfile << " " << S0p << endl;

			//adjust mesh points (change of cohp and Sp)
			myfile << "ADJUST MESH OF PREDICTION" << endl;
			S = Sp;
			S0 = S0p;
			integrate_allcohorts(cohp,out,0);
			if(tempS<=0)
				Sp = 0.000001;
			else
				Sp = tempS;
			cohp = cohnew;
			myfile << "adjusted Sp: " << Sp << endl;
			myfile << "adjusted number of cohorts: " << cohp.size() << endl;
			myfile << "adjusted mesh points:" << endl;
			for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
				myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
			}
			myfile << endl;

			//Newton-correction 

			//calculation of Jacobian J2
			J1 = JacobianGS0(Sp,S0p,cohp,out);
			myfile << " Jacobian J1:" << endl;
			for(int k=0;k!=J1.rows();++k){
				for(int l=0;l!=J1.cols();++l)
					myfile << J1(k,l) << " ";
				myfile << endl;
			}
			myfile << endl;

			//adjust J2 to calculate vp
			J2.setlength(J1.cols(),J1.cols());
			rl.setlength(J1.cols());
			for(int k=0;k<J1.rows();++k){
				for(int l=0;l<J1.cols();++l)
					J2(k,l) = J1(k,l);
				rl(k) = 0;
			}
			rl(J1.rows()) = 1;
			for(int l=0;l<J1.cols();++l)
				J2(J1.rows(),l) = 1;
			alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
			myfile << " vp (unstandardised)=" <<endl;
			for(int k=0;k<J1.cols();++k)
				myfile << v(k) << " ";
			myfile << endl;
			vp = normalize(v);
			myfile << "vp (normalized)=" << endl;
			for(int k=0;k<vp.length();++k)
				myfile << vp(k) << " ";
			myfile << endl;

			//adjust J2 to (J1 vp)^T
			for(int l=0;l<J1.cols();++l)
				J2(J1.rows(),l) = vp(l);

			//calculation of right-hand side R = predicted values - values after 1 map iteration 
			S = Sp;
			S0 = S0p;
			mapM_1(cohp,out);
			R.setlength(J2.cols());
			if(tempS<=0){
				myfile << "values after 1 map iteration (S,b): " << 0.000001;
				R(0) = RCONST(Sp-0.000001);
			}
			else{
				myfile << "values after 1 map iteration (S,b): " << tempS;
				R(0) = RCONST(Sp-tempS);
			}

			myfile << "Sa = " << Sa << ", S0a = " << S0a << endl;
			myfile << endl;

			it = 1;
			cohpit = cohp.begin();
			for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
				myfile << " " << cohit->getb(); 
				R(it) = RCONST(cohpit->getb()-cohit->getb());
				it++;
				cohpit++;
			}
			myfile << endl;
			R(J2.cols()-1) = RCONST(0);
			myfile << "values right-hand side R =";
			normR = RCONST(0);
			for(int l=0;l<R.length();++l){
				myfile << " " << R(l);
				normR += abs(R(l));
			}
			myfile << endl;

			//solve equation J2*cor=R => cor
			alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
			myfile << "values correction = ";
			normcor = RCONST(0);
			for(int l=0;l<cor.length();++l){
				myfile << " " << cor(l);
				normcor += abs(cor(l));
			}
			myfile << endl;
			j++;
			myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
			myfile << "number of Newton steps taken = " << j << endl;
		}

		if((normcor<=VarTolerance)&&(normR<=FunTolerance)){
			//IF SMALL ENOUGH: new fixed point = predicted + correction (with adjusted mesh points)
			myfile << "correction small enough " << endl;

			// adjust prediction:
			Sp += cor(0);
			S0p += cor(cor.length()-1);
			myfile << "adjusted prediction = " << Sp;
			int it = 1;
			for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
				cohpit->setb(RCONST(cohpit->getb()+cor(it)));
				it++;
				myfile << " " << cohpit->getb();
			}
			myfile << " "<< S0p << endl;

			//adjust mesh points (change of cohp and Sp)
			myfile << "ADJUST MESH OF PREDICTION" << endl;
			S = Sp;
			S0 = S0p;
			integrate_allcohorts(cohp,out,0);
			if(tempS<=0)
				Sp = 0.000001;
			else
				Sp = tempS;
			cohp = cohnew;
			myfile << "adjusted Sp (of new fixed point): " << Sp << endl;
			myfile << "adjusted number of cohorts (of new fixed point): " << cohp.size() << endl;
			myfile << "adjusted mesh points (of new fixed point):" << endl;
			for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
				myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
			}
			myfile<< endl;

			Sa = Sp;
			S0a = S0p;
			coh.assign(cohp.begin(),cohp.end());
			p++; //number of calculated fixed points + 1
			myfile << "RESULTS continuation: " << endl;
			myfile << p << " " << Sa << " " << S0a << " " << endl;
			myfile1<< p << " " << Sa << " " << S0a << " " << endl;
			myfile2 << S0a << " " << coh.size() << " " << Sa << " "; 
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
				myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
				btot += cohit->getb();
			}
			myfile << endl;
			myfile1 << endl;
			myfile2 << btot << endl;
			btot = 0;

			//adjust stepsize according to number of Newton steps taken
			if(j>=4){
				step = RCONST(step/2.0); 
				myfile << "more than 4 or 4 Newton steps needed, so stepsize divided by 2" << endl;
				if(step<step_m){
					step = step_m;
					myfile << "minimal Newton step reached" << endl;
				}
				myfile << "new stepsize = " << step << endl;
				myfile << endl << endl;
			}
			else{
				step = RCONST(step*1.3);
				myfile << "less than 4 Newton steps needed, so stepsize * 1.3" << endl;
				if(step>step_M){
					step = step_M;
					myfile << "maximal Newton step reached" << endl;
				}
				myfile << "new stepsize = " << step << endl;
				myfile << endl << endl;
			}
		}

		else{ //more than Newtonmax steps taken, so adjust stepsize and retake prediction
			if(step==step_m){
				myfile << "more than " << Newtonmax << " Newton steps needed and stepsize has reached minimum value " << step_m << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			step = RCONST(step/2.0); 
			if(step<step_m){
				step = step_m;
				myfile << "minimal Newton step reached" << endl;
			}
			myfile << "more than " << Newtonmax << " Newton steps needed, so stepsize/2 for prediction and tried again" << endl;
			myfile << endl;
			q = 1;
		}
	}

	myfile.close();
	return 0;
}

int contfpD(realtype &Sa, realtype &Da, clist &coh, realtype step_start, realtype step_m, realtype step_M, int Newtonmax, realtype FunTolerance,realtype VarTolerance,int dir, int a, int out){ 
	/*start values of fixed point (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + D)
	 parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-001)
	 parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-005)
	 parameter step_M: maximum value of the stepsize of the continuation (eg 1.0)
	 parameter Newtonmax (int): maximum number of Newton steps (eg 10)
	 parameter FunTolerance: precision to be satisfied in Newtoncorrection to have convergence (eg 1.0e-006) for L1-norm of R
	 parameter VarTolerance: precision to be satisfied in Newtoncorrection to have convergence (eg 1.0e-006) for L1-norm of cor
	 parameter dir: gives the direction of the continuation at start -> increasing D if = 1, decreasing D if = -1 
	 parameter a (int): number of fixed points to calculate 
	 parameter out (int): =1 (yes, give full output), = 0 (no, give only important output about the continuation)
	 returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	 changes to original values of Sa, Da and coh made in function!*/

	int j; //number of Newton steps taken
	int p = 0; //number of fixed points calculated
	int q = 0; //used to determine if vector v has to be calculated (=0: yes, =1: no)
	realtype step = step_start; //stepsize continuation
	realtype btot = RCONST(0);

	clistit cohit = coh.begin();
	ofstream myfile;
	ofstream myfile1;
	ofstream myfile2;
	myfile.open("cont.txt", ios::app); 
	myfile.precision(10);
	myfile1.open("continuation.txt", ios::app);
	myfile1.precision(10);
	myfile2.open("cont_DNSb.txt", ios::app);
	myfile2.precision(10);
	myfile << p << " " << Sa << " " << Da << " " << endl;
	myfile1<< p << " " << Sa << " " << Da << " " << endl;
	myfile2<< Da << " " << coh.size() << " " << Sa << " ";
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		btot += cohit->getb();
	}
	myfile << endl;
	myfile1 << endl;
	myfile2 << btot << endl;
	btot = 0;

	if((dir!=1) && (dir!=-1)){
		myfile << "wrong value for the parameter dir: give 1 for increasing D and -1 for decreasing D in the continuation" << endl;
		myfile.close();
		return 1;
	}
	if((out!=1) && (out!=0)){
		myfile << "wrong value for the parameter out: give 1 for full output and 0 for only the important output" << endl;
		myfile.close();
		return 1;
	}	

	clist cohp;
	clistit cohpit;
	cohort b;
	realtype Sp, Dp;
	alglib::real_2d_array JG;
	alglib::real_2d_array JGa;
	alglib::real_1d_array rl;
	alglib::real_1d_array v;
	alglib::real_1d_array vn;
	alglib::real_1d_array vnoud;
	alglib::real_1d_array bp;
	alglib::real_1d_array vp;
	alglib::real_1d_array moud;
	alglib::real_1d_array mnew;
	alglib::ae_int_t info;
	alglib::densesolverreport rep;
	alglib::real_2d_array J1;
	alglib::real_2d_array J2;
	alglib::real_1d_array R; 
	alglib::real_1d_array cor;
	int iadj = 0; //=1 if stepsize for prediction has to be changed 
	realtype normR;
	realtype normcor;

	//FIRST CONTINUATION STEP
	
	//store mass values of cohorts of current fixed point (coh) in mnew
	mnew.setlength(coh.size());
	int it = 0;
	for(cohit=coh.begin();cohit!=coh.end();cohit++){
		mnew(it) = cohit->getMass();
		it++;
	}

	myfile << "calculate Jacobian" << endl;
	myfile << "Sa = " << Sa << ", Da = " << Da << endl;
	JG = JacobianGD(Sa,Da,coh,out);
	myfile << " Jacobian:" << endl;
	for(int k=0;k!=JG.rows();++k){
		for(int l=0;l!=JG.cols();++l)
			myfile << JG(k,l) << " ";
		myfile << endl;
	}
	myfile << endl;

	JGa.setlength(JG.cols(),JG.cols());
	rl.setlength(JG.cols());
	for(int k=0;k<JG.rows();++k){
		for(int l=0;l<JG.cols();++l)
			JGa(k,l) = JG(k,l);
		rl(k) = 0;
	}
	rl(JG.rows()) = 1;
	for(int l=0;l<JG.cols();++l)
		JGa(JG.rows(),l) = 1;
	alglib::rmatrixsolve(JGa,JG.cols(),rl,info,rep,v);
	myfile << " v (unstandardised)=" << endl;
	for(int k=0;k<JG.cols();++k)
		myfile << v(k) << " ";
	myfile << endl;
	vn = normalize(v);
	myfile << "v (normalized)=" << endl;
	for(int k=0;k<vn.length();++k)
		myfile << vn(k) << " ";
	myfile << endl;
		
	if(((vn(vn.length()-1)>0)&&(dir!=1))||((vn(vn.length()-1)<0)&&(dir!=-1))){
		for(int k=0;k<vn.length();++k)
			vn(k) *= -1;
		myfile << "adjust direction of v: yes" << endl;
	}
	else
		myfile << "adjust direction of v: no" << endl;

	// calculation of predicted values:
	Sp = RCONST(Sa+step*vn(0));
	Dp = RCONST(Da+step*vn(vn.length()-1));
	myfile << "predicted values:" << endl;
	myfile << Sp << " " << Dp;
	int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
	bp.setlength(coh.size());
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
		b.set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
		cohp.push_back(b);
		myfile << " " << bp(nc-1);
		nc++;
	}
	myfile << endl;
		
	//adjust stepsize (stepsize/2) because one of the predicted values is <=0? if one of the predicted values is <=0: set (int) iadj=1
	if((Sp<=0)||(Dp<=0))
		iadj = 1;
	else{
		for(int k=0;((k<bp.length())&&(iadj!=1));++k){
			if(bp(k)<=0)
				iadj=1;
		}
	}
		
	while(iadj==1){
		myfile << "one of the predicted values is <=0, this is not allowed, so adjust stepsize (stepsize/2)" << endl;
		if(step==step_m){
			myfile << "stepsize already has reached minimum value " << step_m << endl;
			myfile << "end continuation" << endl;
			return 1;
		}
		step = RCONST(step/2.0);
		if(step<step_m){
			step = step_m;
			myfile << "minimal step reached" << endl;
		}
		myfile << "new stepsize = " << step << endl;
			
		Sp = RCONST(Sa+step*vn(0));
		Dp = RCONST(Da+step*vn(vn.length()-1));
		myfile << "predicted values:" << endl;
		myfile << Sp << " " << Dp;
		cohpit = cohp.begin();
		int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
		bp.setlength(coh.size()); //replace bp
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
			cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
			myfile << " " << cohpit->getb();
			cohpit++;
			nc++;
		}
		myfile << endl;

		iadj = 0;
		if((Sp<=0)||(Dp<=0)){
			iadj = 1;
		}
		else{
			for(int k=0;((k<bp.length())&&(iadj!=1));++k){
				if(bp(k)<=0)
					iadj = 1;
			}
		}
	}

	//adjust mesh points (change of cohp and Sp)
	myfile << "ADJUST MESH OF PREDICTION" << endl;
	S = RCONST(Sp);
	D = RCONST(Dp);
	myfile << "Sa = " << Sa << ", Da = " << Da << endl;
	integrate_allcohorts(cohp,out,0);
	myfile << endl;
	if(tempS <= 0)
		Sp = 0.000001;
	else
		Sp = tempS;
	cohp = cohnew;
	myfile << "adjusted Sp: " << Sp << endl;
	myfile << "adjusted number of cohorts: " << cohp.size() << endl;
	myfile << "adjusted mesh points:" << endl;
	for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
		myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
	}
	myfile << endl;

	//Newton-correction 

	// first step
	j = 1;

	//calculation of Jacobian J1
	J1 = JacobianGD(Sp,Dp,cohp,out);
	myfile << " Jacobian J1:" << endl;
	for(int k=0;k!=J1.rows();++k){
		for(int l=0;l!=J1.cols();++l)
			myfile << J1(k,l) << " ";
		myfile << endl;
	}
	myfile << endl;

	//adjust J1 to J2 to calculate vp
	J2.setlength(J1.cols(),J1.cols());
	rl.setlength(J1.cols());
	for(int k=0;k<J1.rows();++k){
		for(int l=0;l<J1.cols();++l)
			J2(k,l) = J1(k,l);
		rl(k) = 0;
	}
	rl(J1.rows()) = 1;
	for(int l=0;l<J1.cols();++l)
		J2(J1.rows(),l) = 1;
	alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
	myfile << " vp (unstandardised)=" << endl;
	for(int k=0;k<J1.cols();++k)
		myfile << v(k) << " ";
	myfile << endl;
	vp = normalize(v);
	myfile << "vp (normalized)=" << endl;
	for(int k=0;k<vp.length();++k)
		myfile << vp(k) << " ";
	myfile << endl;

	//adjust J2 to (J1 vp)^T
	for(int l=0;l<J1.cols();++l)
		J2(J1.rows(),l) = vp(l);

	myfile << "Sa = " << Sa << ", Da = " << Da << endl;
	myfile << endl;

	//calculation of right-hand side R = predicted values - values after 1 map iteration 
	S = Sp;
	D = Dp;
	mapM_1(cohp,out);
	R.setlength(J2.cols());
	if(tempS<=0){
		myfile << "values after 1 map iteration (S,b): " << 0.000001;
		R(0) = RCONST(Sp-0.000001);
	}
	else{
		myfile << "values after 1 map iteration (S,b): " << tempS;
		R(0) = RCONST(Sp-tempS);
	}
	myfile << endl;

	it = 1;
	cohpit = cohp.begin();
	for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
		myfile << " " << cohit->getb(); 
		R(it) = RCONST(cohpit->getb()-cohit->getb());
		it++;
		cohpit++;
	}
	myfile << endl;
	R(J2.cols()-1) = RCONST(0);
	myfile << "values right-hand side R =";
	normR = RCONST(0);
	for(int l=0;l<R.length();++l){
		myfile << " " << R(l);
		normR += abs(R(l));
	}
	myfile << endl;

	//solve equation J2*cor=R => cor
	alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
	myfile << "values correction = ";
	normcor = RCONST(0);
	for(int l=0;l<cor.length();++l){
		myfile << " " << cor(l);
		normcor += abs(cor(l));
	}
	myfile << endl;
	myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
	myfile << "number of Newton steps taken = " << j << endl;

	// next possible steps

	while( (j<=Newtonmax) && ((normcor>VarTolerance)||(normR>FunTolerance)) ){
		//check if R & cor are small enough (if not: the Newton correction step has to be repeated) (if number of Newton steps taken <= Newtonmax, if more steps are already taken: adjust stepsize and retake prediction)
		myfile << "Newton correction step has to be repeated " << endl;
		// adjust prediction:
		Sp += cor(0);
		Dp += cor(cor.length()-1);
		myfile << "adjusted prediction = " << Sp;
		it = 1;
		for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
			cohpit->setb(RCONST(cohpit->getb()+cor(it)));
			it++;
			myfile << " " << cohpit->getb();
		}
		myfile << " " << Dp << endl;

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		D = Dp;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp: " << Sp << endl;
		myfile << "adjusted number of cohorts: " << cohp.size() << endl;
		myfile << "adjusted mesh points:" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		}
		myfile << endl;

		//Newton-correction 

		//calculation of Jacobian J2
		J1 = JacobianGD(Sp,Dp,cohp,out);
		myfile << " Jacobian J1:" << endl;
		for(int k=0;k!=J1.rows();++k){
			for(int l=0;l!=J1.cols();++l)
				myfile << J1(k,l) << " ";
			myfile << endl;
		}
		myfile << endl;

		//adjust J2 to calculate vp
		J2.setlength(J1.cols(),J1.cols());
		rl.setlength(J1.cols());
		for(int k=0;k<J1.rows();++k){
			for(int l=0;l<J1.cols();++l)
				J2(k,l) = J1(k,l);
			rl(k) = 0;
		}
		rl(J1.rows()) = 1;
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = 1;
		alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
		myfile << " vp (unstandardised)=" << endl;
		for(int k=0;k<J1.cols();++k)
			myfile << v(k) << " ";
		myfile << endl;
		vp = normalize(v);
		myfile << "vp (normalized)=" << endl;
		for(int k=0;k<vp.length();++k)
			myfile << vp(k) << " ";
		myfile << endl;

		//adjust J2 to (J1 vp)^T
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = vp(l);

		//calculation of right-hand side R = predicted values - values after 1 map iteration 
		S = Sp;
		D = Dp;
		mapM_1(cohp,out);
		R.setlength(J2.cols());
		if(tempS<=0){
			myfile << "values after 1 map iteration (S,b): " << 0.000001;
			R(0) = RCONST(Sp-0.000001);
		}
		else{
			myfile << "values after 1 map iteration (S,b): " << tempS;
			R(0) = RCONST(Sp-tempS);
		}

		it = 1;
		cohpit = cohp.begin();
		for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
			myfile << " " << cohit->getb(); 
			R(it) = RCONST(cohpit->getb()-cohit->getb());
			it++;
			cohpit++;
		}
		myfile << endl;
		R(J2.cols()-1) = RCONST(0);
		myfile << "values right-hand side R =";
		normR = RCONST(0);
		for(int l=0;l<R.length();++l){
			myfile << " " << R(l);
			normR += abs(R(l));
		}
		myfile << endl;

		//solve equation J2*cor=R => cor
		alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
		myfile << "values correction = ";
		normcor = RCONST(0);
		for(int l=0;l<cor.length();++l){
			myfile << " " << cor(l);
			normcor += abs(cor(l));
		}
		myfile << endl;
		j++;
		myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
		myfile << "number of Newton steps taken = " << j << endl;
	}
		
	if((normcor<=VarTolerance)&&(normR<=FunTolerance)){
		//IF SMALL ENOUGH: new fixed point = predicted + correction (with adjusted mesh points)
		myfile << "correction small enough " << endl;

		// adjust prediction:
		Sp += cor(0);
		Dp += cor(cor.length()-1);
		myfile << "adjusted prediction = " << Sp;
		int it = 1;
		for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
			cohpit->setb(RCONST(cohpit->getb()+cor(it)));
			it++;
			myfile << " " << cohpit->getb();
		}
		myfile << " " << Dp << endl;

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		D = Dp;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp (of new fixed point): " << Sp << endl;
		myfile << "adjusted number of cohorts (of new fixed point): " << cohp.size() << endl;
		myfile << "adjusted mesh points (of new fixed point):" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		}
		myfile << endl;

		Sa = Sp;
		Da = Dp;
		coh.assign(cohp.begin(),cohp.end());
		p++; //number of calculated fixed points + 1
		myfile << "RESULTS continuation: " << endl;
		myfile << p << " " << Sa << " " << Da << " " << endl;
		myfile1 << p << " " << Sa << " " << Da << " " << endl;
		myfile2 << Da << " " << coh.size() << " " << Sa << " "; 
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			btot += cohit->getb();
		}
		myfile << endl;
		myfile1 << endl;
		myfile2 << btot << endl;
		btot = 0;

		//adjust stepsize according to number of Newton steps taken
		if(j>=4){
			step = RCONST(step/2.0); 
			myfile << "more than 4 or 4 Newton steps needed, so stepsize divided by 2" << endl;
			if(step<step_m){
				step = step_m;
				myfile << "minimal Newton step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			myfile << endl << endl;
		}
		else{
			step = RCONST(step*1.3);
			myfile << "less than 4 Newton steps needed, so stepsize * 1.3" << endl;
			if(step>step_M){
				step = step_M;
				myfile << "maximal Newton step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			myfile << endl <<endl;
		}
	}

	else{ //more than Newtonmax steps taken, so adjust stepsize and retake prediction
		if(step==step_m){
			myfile << "more than " << Newtonmax << " Newton steps needed and stepsize has reached minimum value " << step_m << endl;
			myfile << "end continuation" << endl;
			return 1;
		}
		step = RCONST(step/2.0); 
		if(step<step_m){
			step = step_m;
			myfile << "minimal Newton step reached" << endl;
		}
		myfile << "more than " << Newtonmax << " Newton steps needed, so stepsize/2 for prediction and tried again" << endl;
		myfile << endl;
		q = 1;
	}

	// LOOP OF NEXT CONTINUATION STEPS (difference with first one: dot product must be positive with tangent vector of previous step to determine direction of new tangent vector for prediction)
	while(p<a){ //loop of calculation of fixed points
		myfile << "q = " << q << endl;
		if(q==0){
			//store mass values of cohorts of previous fixed point in moud (mnew=>moud)
			moud.setlength(mnew.length());
			for(int k=0;k!=mnew.length();++k)
				moud(k) = mnew(k);
			//store mass values of cohorts of current fixed point (coh) in mnew
			mnew.setlength(coh.size());
			it = 0;
			for(cohit=coh.begin();cohit!=coh.end();cohit++){
				mnew(it) = cohit->getMass();
				it++;
			}
			//store tangent vector at previous fixed point (vn) in vnoud
			vnoud.setlength(vn.length());
			for(int k=0;k!=vn.length();++k)
				vnoud(k) = vn(k);

			JG = JacobianGD(Sa,Da,coh,out);
			myfile << " Jacobian:" << endl;
			for(int k=0;k!=JG.rows();++k){
				for(int l=0;l!=JG.cols();++l)
					myfile << JG(k,l) << " ";
				myfile << endl;
			}
			myfile << endl;

			JGa.setlength(JG.cols(),JG.cols());
			rl.setlength(JG.cols());
			for(int k=0;k<JG.rows();++k){
				for(int l=0;l<JG.cols();++l)
					JGa(k,l) = JG(k,l);
				rl(k) = 0;
			}
			rl(JG.rows()) = 1;
			for(int l=0;l<JG.cols();++l)
				JGa(JG.rows(),l) = 1;
			alglib::rmatrixsolve(JGa,JG.cols(),rl,info,rep,v);
			myfile << " v (unstandardised)=" << endl;
			for(int k=0;k<JG.cols();++k)
				myfile << v(k) << " ";
			myfile << endl;
			vn = normalize(v);
			myfile << "v (normalized)=" << endl;
			for(int k=0;k<vn.length();++k)
				myfile << vn(k) << " ";
			myfile << endl;

			//check direction of vn: dot product with vnoud must be positive, otherwise: adjust direction
			if(signdotproduct(vnoud,vn,moud,mnew,out)==0){
				myfile << "problem with sizes of vectors for signdotproduct" << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			else{
				if(signdotproduct(vnoud,vn,moud,mnew,out)==+1)
					myfile << "adjust direction of v: no" << endl;
				else{
					if(signdotproduct(vnoud,vn,moud,mnew,out)==-1){
						myfile << "adjust direction of v: yes" << endl;
						for(int k=0;k<vn.length();++k)
							vn(k) *= -1;
					}
					else{
						myfile << "problem with signdotproduct" << endl;
						myfile << "end continuation" << endl;
						return 1;
					}
				}
			}

			/*if(vn(0)>=0){
				myfile << "adjust direction of v: yes" << endl;
				for(int k=0;k<vn.length();++k)
					vn(k) *= -1;
			}*/
		}
		q = 0;

		// calculation of predicted values:
		myfile << "step = " << step << ", Sa = " << Sa << ", Da = " << Da << endl;
		myfile << "vn = " << endl;
		for(int k=0; k<vn.length(); ++k)
			myfile << vn(k) << " ";
		myfile << endl;
		myfile << "coh = " << endl;
		for(cohit=coh.begin();cohit!=coh.end();++cohit){ 
			myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		}
		myfile << endl;

		Sp = RCONST(Sa+step*vn(0));
		Dp = RCONST(Da+step*vn(vn.length()-1));
		myfile << "predicted values:" << endl;
		myfile << Sp << " " << Dp;
		cohp.assign(coh.begin(),coh.end());
		cohpit = cohp.begin();
		int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
		bp.setlength(coh.size());
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
			cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
			myfile << " " << cohpit->getb();
			cohpit++;
			nc++;
		}
		myfile << endl;
		
		//adjust stepsize (stepsize/2) because one of the predicted values is <=0? if one of the predicted values is <=0: set (int) iadj=1
		iadj = 0;
		if((Sp<=0)||(Dp<=0))
			iadj = 1;
		else{
			for(int k=0;((k<bp.length())&&(iadj!=1));++k){
				if(bp(k)<=0)
					iadj = 1;
			}
		}
		
		while(iadj==1){
			myfile << "one of the predicted values is <=0, this is not allowed, so adjust stepsize (stepsize/2)" << endl;
			if(step==step_m){
				myfile << "stepsize already has reached minimum value " << step_m << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			step = RCONST(step/2.0);
			if(step<step_m){
				step = step_m;
				myfile << "minimal step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			
			Sp = RCONST(Sa+step*vn(0));
			Dp = RCONST(Da+step*vn(vn.length()-1));
			myfile << "predicted values:" << endl;
			myfile << Sp << " " << Dp;
			cohp.assign(coh.begin(),coh.end());
			cohpit = cohp.begin();
			int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
			bp.setlength(coh.size());
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
				cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
				myfile << " " << cohpit->getb();
				cohpit++;
				nc++;
			}
			myfile << endl;
			iadj = 0;
			if((Sp<=0)||(Dp<=0))
				iadj = 1;
			else{
				for(int k=0;((k<bp.length())&&(iadj!=1));++k){
					if(bp(k)<=0)
						iadj = 1;
				}
			}
		}

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		D = Dp;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp: " << Sp << endl;
		myfile << "adjusted number of cohorts: " << cohp.size() << endl;
		myfile << "adjusted mesh points:" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		}
		myfile << endl;

		//Newton-correction 

		// first step
		j = 1;

		//calculation of Jacobian J2
		J1 = JacobianGD(Sp,Dp,cohp,out);
		myfile << " Jacobian J1:" << endl;
		for(int k=0;k!=J1.rows();++k){
			for(int l=0;l!=J1.cols();++l)
				myfile << J1(k,l) << " ";
			myfile << endl;
		}
		myfile << endl;

		myfile << "Sa = " << Sa << ", Da = " << Da << endl;
		myfile << endl;

		//adjust J2 to calculate vp
		J2.setlength(J1.cols(),J1.cols());
		rl.setlength(J1.cols());
		for(int k=0;k<J1.rows();++k){
			for(int l=0;l<J1.cols();++l)
				J2(k,l) = J1(k,l);
			rl(k) = 0;
		}
		rl(J1.rows()) = 1;
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = 1;
		alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
		myfile << " vp (unstandardised)=" << endl;
		for(int k=0;k<J1.cols();++k)
			myfile << v(k) << " ";
		myfile << endl;
		vp = normalize(v);
		myfile << "vp (normalized)=" << endl;
		for(int k=0;k<vp.length();++k)
			myfile << vp(k) << " ";
		myfile << endl;

		//adjust J2 to (J1 vp)^T
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = vp(l);

		myfile << "Sa = " << Sa << ", Da = " << Da << endl;
		myfile << endl;

		//calculation of right-hand side R = predicted values - values after 1 map iteration 
		S = Sp;
		D = Dp;
		mapM_1(cohp,out);
		R.setlength(J2.cols());
		if(tempS<=0){
			myfile << "values after 1 map iteration (S,b): " << 0.000001;
			R(0) = RCONST(Sp-0.000001);
		}
		else{
			myfile << "values after 1 map iteration (S,b): " << tempS;
			R(0) = RCONST(Sp-tempS);
		}

		myfile << "Sa = " << Sa << ", Da = " << Da << endl;
		myfile << endl;

		int it = 1;
		cohpit = cohp.begin();
		for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
			myfile << " " << cohit->getb(); 
			R(it) = RCONST(cohpit->getb()-cohit->getb());
			it++;
			cohpit++;
		}
		myfile << endl;
		R(J2.cols()-1) = RCONST(0);
		myfile << "values right-hand side R =";
		normR = RCONST(0);
		for(int l=0;l<R.length();++l){
			myfile << " " << R(l);
			normR += abs(R(l));
		}
		myfile << endl;

		//solve equation J2*cor=R => cor
		alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
		myfile << "values correction = ";
		normcor = RCONST(0);
		for(int l=0;l<cor.length();++l){
			myfile << " " << cor(l);
			normcor += abs(cor(l));
		}
		myfile << endl;
		myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
		myfile << "number of Newton steps taken = " << j << endl;

		// next possible steps

		while((j<=Newtonmax)&&((normcor>VarTolerance)||(normR>FunTolerance))){

			//check if R & cor are small enough (if not: the Newton correction step has to be repeated) (if number of Newton steps taken <= Newtonmax, if more steps are already taken: adjust stepsize and retake prediction)
			myfile << "Newton correction step has to be repeated " << endl;
			// adjust prediction:
			Sp += cor(0);
			Dp += cor(cor.length()-1);
			myfile << "adjusted prediction = " << Sp;
			it = 1;
			for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
				cohpit->setb(RCONST(cohpit->getb()+cor(it)));
				it++;
				myfile << " " << cohpit->getb();
			}
			myfile << " " << Dp << endl;

			//adjust mesh points (change of cohp and Sp)
			myfile << "ADJUST MESH OF PREDICTION" << endl;
			S = Sp;
			D = Dp;
			integrate_allcohorts(cohp,out,0);
			if(tempS<=0)
				Sp = 0.000001;
			else
				Sp = tempS;
			cohp = cohnew;
			myfile << "adjusted Sp: " << Sp << endl;
			myfile << "adjusted number of cohorts: " << cohp.size() << endl;
			myfile << "adjusted mesh points:" << endl;
			for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit)
				myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
			myfile << endl;

			//Newton-correction 

			//calculation of Jacobian J2
			J1 = JacobianGD(Sp,Dp,cohp,out);
			myfile << " Jacobian J1:" << endl;
			for(int k=0;k!=J1.rows();++k){
				for(int l=0;l!=J1.cols();++l)
					myfile << J1(k,l) << " ";
				myfile << endl;
			}
			myfile << endl;

			//adjust J2 to calculate vp
			J2.setlength(J1.cols(),J1.cols());
			rl.setlength(J1.cols());
			for(int k=0;k<J1.rows();++k){
				for(int l=0;l<J1.cols();++l)
					J2(k,l) = J1(k,l);
				rl(k) = 0;
			}
			rl(J1.rows()) = 1;
			for(int l=0;l<J1.cols();++l)
				J2(J1.rows(),l) = 1;
			alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
			myfile << " vp (unstandardised)=" << endl;
			for(int k=0;k<J1.cols();++k)
				myfile << v(k) << " ";
			myfile << endl;
			vp = normalize(v);
			myfile << "vp (normalized)=" << endl;
			for(int k=0;k<vp.length();++k)
				myfile << vp(k) << " ";
			myfile << endl;

			//adjust J2 to (J1 vp)^T
			for(int l=0;l<J1.cols();++l)
				J2(J1.rows(),l) = vp(l);

			//calculation of right-hand side R = predicted values - values after 1 map iteration 
			S = Sp;
			D = Dp;
			mapM_1(cohp,out);
			R.setlength(J2.cols());
			if(tempS<=0){
				myfile << "values after 1 map iteration (S,b): " << 0.000001;
				R(0) = RCONST(Sp-0.000001);
			}
			else{
				myfile << "values after 1 map iteration (S,b): " << tempS;
				R(0) = RCONST(Sp-tempS);
			}

			myfile << "Sa = " << Sa << ", Da = " << Da << endl;
			myfile << endl;

			it = 1;
			cohpit = cohp.begin();
			for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
				myfile << " " << cohit->getb(); 
				R(it) = RCONST(cohpit->getb()-cohit->getb());
				it++;
				cohpit++;
			}
			myfile << endl;
			R(J2.cols()-1) = RCONST(0);
			myfile << "values right-hand side R =";
			normR = RCONST(0);
			for(int l=0;l<R.length();++l){
				myfile << " " << R(l);
				normR += abs(R(l));
			}
			myfile << endl;

			//solve equation J2*cor=R => cor
			alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
			myfile << "values correction = ";
			normcor = RCONST(0);
			for(int l=0;l<cor.length();++l){
				myfile << " " << cor(l);
				normcor += abs(cor(l));
			}
			myfile << endl;
			j++;
			myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
			myfile << "number of Newton steps taken = " << j << endl;
		}
		
		if((normcor<=VarTolerance)&&(normR<=FunTolerance)){
			//IF SMALL ENOUGH: new fixed point = predicted + correction (with adjusted mesh points)
			myfile << "correction small enough " << endl;

			// adjust prediction:
			Sp += cor(0);
			Dp += cor(cor.length()-1);
			myfile << "adjusted prediction = " << Sp;
			int it = 1;
			for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
				cohpit->setb(RCONST(cohpit->getb()+cor(it)));
				it++;
				myfile << " " << cohpit->getb();
			}
			myfile << " " << Dp << endl;

			//adjust mesh points (change of cohp and Sp)
			myfile << "ADJUST MESH OF PREDICTION" << endl;
			S = Sp;
			D = Dp;
			integrate_allcohorts(cohp,out,0);
			if(tempS<=0)
				Sp = 0.000001;
			else
				Sp = tempS;
			cohp = cohnew;
			myfile << "adjusted Sp (of new fixed point): " << Sp << endl;
			myfile << "adjusted number of cohorts (of new fixed point): " << cohp.size() << endl;
			myfile << "adjusted mesh points (of new fixed point):" << endl;
			for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
				myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
			}
			myfile << endl;

			Sa = Sp;
			Da = Dp;
			coh.assign(cohp.begin(),cohp.end());
			p++; //number of calculated fixed points + 1
			myfile << "RESULTS continuation: " << endl;
			myfile << p << " " << Sa << " " << Da << " " << endl;
			myfile1<< p << " " << Sa << " " << Da << " " << endl;
			myfile2 << Da << " " << coh.size() << " " << Sa << " "; 
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
				myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
				btot += cohit->getb();
			}
			myfile << endl;
			myfile1 << endl;
			myfile2 << btot << endl;
			btot = 0;

			//adjust stepsize according to number of Newton steps taken
			if(j>=4){
				step = RCONST(step/2.0); 
				myfile << "more than 4 or 4 Newton steps needed, so stepsize divided by 2" << endl;
				if(step<step_m){
					step = step_m;
					myfile << "minimal Newton step reached" << endl;
				}
				myfile << "new stepsize = " << step << endl;
				myfile << endl << endl;
			}
			else{
				step = RCONST(step*1.3);
				myfile << "less than 4 Newton steps needed, so stepsize * 1.3" << endl;
				if(step>step_M){
					step = step_M;
					myfile << "maximal Newton step reached" << endl;
				}
				myfile << "new stepsize = " << step << endl;
				myfile << endl << endl;
			}
		}

		else{ //more than Newtonmax steps taken, so adjust stepsize and retake prediction
			if(step==step_m){
				myfile << "more than " << Newtonmax << " Newton steps needed and stepsize has reached minimum value " << step_m << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			step = RCONST(step/2.0); 
			if(step<step_m){
				step = step_m;
				myfile << "minimal Newton step reached" << endl;
			}
			myfile << "more than " << Newtonmax << " Newton steps needed, so stepsize/2 for prediction and tried again" << endl;
			myfile << endl;
			q = 1;
		}
	}

	myfile.close();
	return 0;
}

int contfpD_demp(realtype &Sa, realtype &Da, clist &coh, realtype step_start, realtype step_m, realtype step_M, int Newtonmax, realtype demping, realtype FunTolerance,realtype VarTolerance,int dir, int a, int out){ 
	/*start values of fixed point (m,X,Y and A and number of cells in birth cohorts in list of cohorts coh + S + D)
	 parameter step_start: the stepsize of the continuation at the start of the continuation (eg 1.0e-001)
	 parameter step_m: minimum value of the stepsize of the continuation (eg 1.0e-005)
	 parameter step_M: maximum value of the stepsize of the continuation (eg 1.0)
	 parameter Newtonmax (int): maximum number of Newton steps (eg 10)
	 parameter demping: percentage of Newton step that is taken (eg 0.25 means 25% of the Newton correction)
	 parameter FunTolerance: precision to be satisfied in Newtoncorrection to have convergence (eg 1.0e-006) for L1-norm of R
	 parameter VarTolerance: precision to be satisfied in Newtoncorrection to have convergence (eg 1.0e-006) for L1-norm of cor
	 parameter dir: gives the direction of the continuation at start -> increasing D if = 1, decreasing D if = -1 
	 parameter a (int): number of fixed points to calculate 
	 parameter out (int): =1 (yes, give full output), = 0 (no, give only important output about the continuation)
	 returns 1 if something went wrong (see error in cont.txt) and 0 otherwise
	 changes to original values of Sa, Da and coh made in function!*/

	int j; //number of Newton steps taken
	int p = 0; //number of fixed points calculated
	int q = 0; //used to determine if vector v has to be calculated (=0: yes, =1: no)
	realtype step = step_start; //stepsize continuation
	realtype btot = RCONST(0);

	clistit cohit = coh.begin();
	ofstream myfile;
	ofstream myfile1;
	ofstream myfile2;
	myfile.open ("cont.txt", ios::app); 
	myfile.precision(10);
	myfile1.open("continuation.txt", ios::app);
	myfile1.precision(10);
	myfile2.open("cont_DNSb.txt", ios::app);
	myfile2.precision(10);
	myfile << p << " " << Sa << " " << Da << " " << endl;
	myfile1<< p << " " << Sa << " " << Da << " " << endl;
	myfile2<< Da << " " << coh.size() << " " << Sa << " ";
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		btot += cohit->getb();
	}
	myfile << endl;
	myfile1 << endl;
	myfile2 << btot << endl;
	btot = 0;

	if((dir!=1) && (dir!=-1)){
		myfile << "wrong value for the parameter dir: give 1 for increasing D and -1 for decreasing D in the continuation" << endl;
		myfile.close();
		return 1;
	}
	if((out!=1) && (out!=0)){
		myfile << "wrong value for the parameter out: give 1 for full output and 0 for only the important output" << endl;
		myfile.close();
		return 1;
	}	

	clist cohp;
	clistit cohpit;
	cohort b;
	realtype Sp, Dp;
	alglib::real_2d_array JG;
	alglib::real_2d_array JGa;
	alglib::real_1d_array rl;
	alglib::real_1d_array v;
	alglib::real_1d_array vn;
	alglib::real_1d_array vnoud;
	alglib::real_1d_array bp;
	alglib::real_1d_array vp;
	alglib::real_1d_array moud;
	alglib::real_1d_array mnew;
	alglib::ae_int_t info;
	alglib::densesolverreport rep;
	alglib::real_2d_array J1;
	alglib::real_2d_array J2;
	alglib::real_1d_array R; 
	alglib::real_1d_array cor;
	int iadj = 0; //=1 if stepsize for prediction has to be changed 
	realtype normR;
	realtype normcor;
	
	//FIRST CONTINUATION STEP
	
	//store mass values of cohorts of current fixed point (coh) in mnew
	mnew.setlength(coh.size());
	int it = 0;
	for(cohit=coh.begin();cohit!=coh.end();cohit++){
		mnew(it) = cohit->getMass();
		it++;
	}

	myfile << "calculate Jacobian" << endl;
	myfile << "Sa = " << Sa << ", Da = " << Da << endl;
	JG = JacobianGD(Sa,Da,coh,out);
	myfile << " Jacobian:" << endl;
	for(int k=0;k!=JG.rows();++k){
		for(int l=0;l!=JG.cols();++l)
			myfile << JG(k,l) << " ";
		myfile << endl;
	}
	myfile << endl;

	JGa.setlength(JG.cols(),JG.cols());
	rl.setlength(JG.cols());
	for(int k=0;k<JG.rows();++k){
		for(int l=0;l<JG.cols();++l)
			JGa(k,l) = JG(k,l);
		rl(k) = 0;
	}
	rl(JG.rows()) = 1;
	for(int l=0;l<JG.cols();++l)
		JGa(JG.rows(),l) = 1;
	alglib::rmatrixsolve(JGa,JG.cols(),rl,info,rep,v);
	myfile << " v (unstandardised)=" << endl;
	for(int k=0;k<JG.cols();++k)
		myfile << v(k) << " ";
	myfile << endl;
	vn = normalize(v);
	myfile << "v (normalized)=" << endl;
	for(int k=0;k<vn.length();++k)
		myfile << vn(k) << " ";
	myfile << endl;
		
	if(((vn(vn.length()-1)>0)&&(dir!=1))||((vn(vn.length()-1)<0)&&(dir!=-1))){
		for(int k=0;k<vn.length();++k)
			vn(k) *= -1;
		myfile << "adjust direction of v: yes" << endl;
	}
	else{
		myfile << "adjust direction of v: no" << endl;
	}

	// calculation of predicted values:
	Sp = RCONST(Sa+step*vn(0));
	Dp = RCONST(Da+step*vn(vn.length()-1));
	myfile << "predicted values:" << endl;
	myfile << Sp << " " << Dp;
	int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
	bp.setlength(coh.size());
	for(cohit=coh.begin();cohit!=coh.end();++cohit){
		bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
		b.set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
		cohp.push_back(b);
		myfile << " " << bp(nc-1);
		nc++;
	}
	myfile << endl;
		
	//adjust stepsize (stepsize/2) because one of the predicted values is <=0? if one of the predicted values is <=0: set (int) iadj=1
	if((Sp<=0)||(Dp<=0))
		iadj = 1;
	else{
		for(int k=0;((k<bp.length())&&(iadj!=1));++k){
			if(bp(k)<=0)
				iadj = 1;
		}
	}
		
	while(iadj==1){
		myfile << "one of the predicted values is <=0, this is not allowed, so adjust stepsize (stepsize/2)" << endl;
		if(step==step_m){
			myfile << "stepsize already has reached minimum value " << step_m << endl;
			myfile << "end continuation" << endl;
			return 1;
		}
		step = RCONST(step/2.0);
		if(step<step_m){
			step = step_m;
			myfile << "minimal step reached" << endl;
		}
		myfile << "new stepsize = " << step << endl;
			
		Sp = RCONST(Sa+step*vn(0));
		Dp = RCONST(Da+step*vn(vn.length()-1));
		myfile << "predicted values:" << endl;
		myfile << Sp << " " << Dp;
		cohpit = cohp.begin();
		int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
		bp.setlength(coh.size()); //replace bp
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
			cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
			myfile << " " << cohpit->getb();
			cohpit++;
			nc++;
		}
		myfile << endl;
		iadj = 0;
		if((Sp<=0)||(Dp<=0))
			iadj = 1;
		else{
			for(int k=0;((k<bp.length())&&(iadj!=1));++k){
				if(bp(k)<=0)
					iadj = 1;
			}
		}
	}

	//adjust mesh points (change of cohp and Sp)
	myfile << "ADJUST MESH OF PREDICTION" << endl;
	S = RCONST(Sp);
	D = RCONST(Dp);
	myfile << "Sa = " << Sa << ", Da = " << Da << endl;
	integrate_allcohorts(cohp,out,0);
	myfile << endl;
	if(tempS <= 0)
		Sp = 0.000001;
	else
		Sp = tempS;
	cohp = cohnew;
	myfile << "adjusted Sp: " << Sp << endl;
	myfile << "adjusted number of cohorts: " << cohp.size() << endl;
	myfile << "adjusted mesh points:" << endl;
	for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
		myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
	}
	myfile << endl;

	//Newton-correction 

	// first step
	j = 1;

	//calculation of Jacobian J1
	J1 = JacobianGD(Sp,Dp,cohp,out);
	myfile << " Jacobian J1:" << endl;
	for(int k=0;k!=J1.rows();++k){
		for(int l=0;l!=J1.cols();++l)
			myfile << J1(k,l) << " ";
		myfile << endl;
	}
	myfile << endl;

	//adjust J1 to J2 to calculate vp
	J2.setlength(J1.cols(),J1.cols());
	rl.setlength(J1.cols());
	for(int k=0;k<J1.rows();++k){
		for(int l=0;l<J1.cols();++l)
			J2(k,l) = J1(k,l);
		rl(k) = 0;
	}
	rl(J1.rows()) = 1;
	for(int l=0;l<J1.cols();++l)
		J2(J1.rows(),l) = 1;
	alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
	myfile << " vp (unstandardised)=" << endl;
	for(int k=0;k<J1.cols();++k)
		myfile << v(k) << " ";
	myfile << endl;
	vp = normalize(v);
	myfile << "vp (normalized)=" << endl;
	for(int k=0;k<vp.length();++k)
		myfile << vp(k) << " ";
	myfile << endl;

	//adjust J2 to (J1 vp)^T
	for(int l=0;l<J1.cols();++l)
		J2(J1.rows(),l) = vp(l);

	//calculation of right-hand side R = predicted values - values after 1 map iteration 
	S = Sp;
	D = Dp;
	mapM_1(cohp,out);
	R.setlength(J2.cols());
	if(tempS<=0){
		myfile << "values after 1 map iteration (S,b): " << 0.000001;
		R(0) = RCONST(Sp-0.000001);
	}
	else{
		myfile << "values after 1 map iteration (S,b): " << tempS;
		R(0) = RCONST(Sp-tempS);
	}

	it = 1;
	cohpit = cohp.begin();
	for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
		myfile << " " << cohit->getb(); 
		R(it) = RCONST(cohpit->getb()-cohit->getb());
		it++;
		cohpit++;
	}
	myfile << endl;
	R(J2.cols()-1) = RCONST(0);
	myfile << "values right-hand side R =";
	normR = RCONST(0);
	for(int l=0;l<R.length();++l){
		myfile << " " << R(l);
		normR += abs(R(l));
	}
	myfile << endl;

	//solve equation J2*cor=R => cor
	alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
	myfile << "values correction = ";
	normcor = RCONST(0);
	for(int l=0;l<cor.length();++l){
		myfile << " " << cor(l);
		normcor += abs(cor(l));
	}
	myfile << endl;
	myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
	myfile << "number of Newton steps taken = " << j << endl;

	// next possible steps
	while( (j<=Newtonmax) && ((normcor>VarTolerance)||(normR>FunTolerance)) ){

		//check if R & cor are small enough (if not: the Newton correction step has to be repeated) (if number of Newton steps taken <= Newtonmax, if more steps are already taken: adjust stepsize and retake prediction)
		myfile << "Newton correction step has to be repeated " << endl;
		// adjust prediction:
		Sp += demping*cor(0);
		Dp += demping*cor(cor.length()-1);
		myfile << "adjusted prediction = " << Sp;
		it = 1;
		for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
			cohpit->setb(RCONST(cohpit->getb()+demping*cor(it)));
			it++;
			myfile << " " << cohpit->getb();
		}
		myfile << " " << Dp << endl;

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		D = Dp;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp: " << Sp << endl;
		myfile << "adjusted number of cohorts: " << cohp.size() << endl;
		myfile << "adjusted mesh points:" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		}
		myfile << endl;

		//Newton-correction 

		//calculation of Jacobian J2
		J1 = JacobianGD(Sp,Dp,cohp,out);
		myfile << " Jacobian J1:" << endl;
		for(int k=0;k!=J1.rows();++k){
			for(int l=0;l!=J1.cols();++l)
				myfile << J1(k,l) << " ";
			myfile << endl;
		}
		myfile << endl;

		//adjust J2 to calculate vp
		J2.setlength(J1.cols(),J1.cols());
		rl.setlength(J1.cols());
		for(int k=0;k<J1.rows();++k){
			for(int l=0;l<J1.cols();++l)
				J2(k,l) = J1(k,l);
			rl(k) = 0;
		}
		rl(J1.rows()) = 1;
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = 1;
		alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
		myfile << " vp (unstandardised)=" << endl;
		for(int k=0;k<J1.cols();++k)
			myfile << v(k) << " ";
		myfile << endl;
		vp = normalize(v);
		myfile << "vp (normalized)=" << endl;
		for(int k=0;k<vp.length();++k)
			myfile << vp(k) << " ";
		myfile << endl;

		//adjust J2 to (J1 vp)^T
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = vp(l);

		//calculation of right-hand side R = predicted values - values after 1 map iteration 
		S = Sp;
		D = Dp;
		mapM_1(cohp,out);
		R.setlength(J2.cols());
		if(tempS<=0){
			myfile << "values after 1 map iteration (S,b): " << 0.000001;
			R(0) = RCONST(Sp-0.000001);
		}
		else{
			myfile << "values after 1 map iteration (S,b): " << tempS;
			R(0) = RCONST(Sp-tempS);
		}

		it = 1;
		cohpit = cohp.begin();
		for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
			myfile << " " << cohit->getb(); 
			R(it) = RCONST(cohpit->getb()-cohit->getb());
			it++;
			cohpit++;
		}
		myfile << endl;
		R(J2.cols()-1) = RCONST(0);
		myfile << "values right-hand side R =";
		normR = RCONST(0);
		for(int l=0;l<R.length();++l){
			myfile << " " << R(l);
			normR += abs(R(l));
		}
		myfile << endl;

		//solve equation J2*cor=R => cor
		alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
		myfile << "values correction = ";
		normcor = RCONST(0);
		for(int l=0;l<cor.length();++l){
			myfile << " " << cor(l);
			normcor += abs(cor(l));
		}
		myfile << endl;
		j++;
		myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance << ")" << endl;
		myfile << "number of Newton steps taken = " << j << endl;
	}
		
	if((normcor<=VarTolerance)&&(normR<=FunTolerance)){
		//IF SMALL ENOUGH: new fixed point = predicted + correction (with adjusted mesh points)
		myfile << "correction small enough " << endl;

		// adjust prediction:
		Sp += demping*cor(0);
		Dp += demping*cor(cor.length()-1);
		myfile << "adjusted prediction = " << Sp;
		int it = 1;
		for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
			cohpit->setb(RCONST(cohpit->getb()+demping*cor(it)));
			it++;
			myfile << " " << cohpit->getb();
		}
		myfile << " " << Dp << endl;

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		D = Dp;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp (of new fixed point): " << Sp << endl;
		myfile << "adjusted number of cohorts (of new fixed point): " << cohp.size() << endl;
		myfile << "adjusted mesh points (of new fixed point):" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		}
		myfile << endl;

		Sa = Sp;
		Da = Dp;
		coh.assign(cohp.begin(),cohp.end());
		p++; //number of calculated fixed points + 1
		myfile << "RESULTS continuation: " << endl;
		myfile << p << " " << Sa << " " << Da << " " << endl;
		myfile1 << p << " " << Sa << " " << Da << " " << endl;
		myfile2 << Da << " " << coh.size() << " " << Sa << " "; 
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
			btot += cohit->getb();
		}
		myfile << endl;
		myfile1 << endl;
		myfile2 << btot << endl;
		btot = 0;

		myfile << "Sa = " << Sa << ", Da = " << Da << endl;
		myfile << endl;

		//adjust stepsize according to number of Newton steps taken
		if(j>=4){
			step = RCONST(step/2.0); 
			myfile << "more than 4 or 4 Newton steps needed, so stepsize divided by 2" << endl;
			if(step<step_m){
				step = step_m;
				myfile << "minimal Newton step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			myfile << endl << endl;
		}
		else{
			step = RCONST(step*1.3);
			myfile << "less than 4 Newton steps needed, so stepsize * 1.3" << endl;
			if(step>step_M){
				step = step_M;
				myfile << "maximal Newton step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			myfile << endl << endl;
		}
	}

	else{ //more than Newtonmax steps taken, so adjust stepsize and retake prediction
		if(step==step_m){
			myfile << "more than " << Newtonmax << " Newton steps needed and stepsize has reached minimum value " << step_m << endl;
			myfile << "end continuation" << endl;
			return 1;
		}
		step = RCONST(step/2.0); 
		if(step<step_m){
			step = step_m;
			myfile << "minimal Newton step reached" << endl;
		}
		myfile << "more than " << Newtonmax << " Newton steps needed, so stepsize/2 for prediction and tried again" << endl;
		myfile << endl;
		q = 1;
	}

	// LOOP OF NEXT CONTINUATION STEPS (difference with first one: dot product must be positive with tangent vector of previous step to determine direction of new tangent vector for prediction)
	while(p<a){ //loop of calculation of fixed points
		myfile << "q = " << q << endl;
		if(q==0){
			//store mass values of cohorts of previous fixed point in moud (mnew=>moud)
			moud.setlength(mnew.length());
			for(int k=0;k!=mnew.length();++k)
				moud(k) = mnew(k);
			//store mass values of cohorts of current fixed point (coh) in mnew
			mnew.setlength(coh.size());
			it=0;
			for(cohit=coh.begin();cohit!=coh.end();cohit++){
				mnew(it) = cohit->getMass();
				it++;
			}
			//store tangent vector at previous fixed point (vn) in vnoud
			vnoud.setlength(vn.length());
			for(int k=0;k!=vn.length();++k)
				vnoud(k) = vn(k);

			JG = JacobianGD(Sa,Da,coh,out);
			myfile << " Jacobian:" << endl;
			for(int k=0;k!=JG.rows();++k){
				for(int l=0;l!=JG.cols();++l)
					myfile << JG(k,l) << " ";
				myfile << endl;
			}
			myfile << endl;

			JGa.setlength(JG.cols(),JG.cols());
			rl.setlength(JG.cols());
			for(int k=0;k<JG.rows();++k){
				for(int l=0;l<JG.cols();++l)
					JGa(k,l) = JG(k,l);
				rl(k) = 0;
			}
			rl(JG.rows()) = 1;
			for(int l=0;l<JG.cols();++l)
				JGa(JG.rows(),l) = 1;
			alglib::rmatrixsolve(JGa,JG.cols(),rl,info,rep,v);
			myfile << " v (unstandardised)=" << endl;
			for(int k=0;k<JG.cols();++k)
				myfile << v(k) << " ";
			myfile << endl;
			vn = normalize(v);
			myfile << "v (normalized)=" << endl;
			for(int k=0;k<vn.length();++k)
				myfile << vn(k) << " ";
			myfile << endl;

			//check direction of vn: dot product with vnoud must be positive, otherwise: adjust direction
			if(signdotproduct(vnoud,vn,moud,mnew,out)==0){
				myfile << "problem with sizes of vectors for signdotproduct" <<endl;
				myfile << "end continuation" <<endl;
				return 1;
			}
			else{
				if(signdotproduct(vnoud,vn,moud,mnew,out)==+1)
					myfile << "adjust direction of v: no" <<endl;
				else{
					if(signdotproduct(vnoud,vn,moud,mnew,out)==-1){
						myfile << "adjust direction of v: yes" <<endl;
						for(int k=0;k<vn.length();++k)
							vn(k)*=-1;
					}
					else{
						myfile << "problem with signdotproduct" <<endl;
						myfile << "end continuation" <<endl;
						return 1;
					}
				}
			}

			/*if(vn(0)>=0){
				myfile << "adjust direction of v: yes" <<endl;
				for(int k=0;k<vn.length();++k)
					vn(k)*=-1;
			}*/
		}
		q = 0;

		// calculation of predicted values:
		myfile << "step = " << step << ", Sa = " << Sa << ", Da = " << Da << endl;
		myfile << "vn = " << endl;
		for(int k=0; k<vn.length(); ++k)
			myfile << vn(k) << " ";
		myfile << endl;
		myfile << "coh = " << endl;
		for(cohit=coh.begin();cohit!=coh.end();++cohit){ 
			myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
		}
		myfile << endl;

		Sp = RCONST(Sa+step*vn(0));
		Dp = RCONST(Da+step*vn(vn.length()-1));
		myfile << "predicted values:" << endl;
		myfile << Sp << " " << Dp;
		cohp.assign(coh.begin(),coh.end());
		cohpit = cohp.begin();
		int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
		bp.setlength(coh.size());
		for(cohit=coh.begin();cohit!=coh.end();++cohit){
			bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
			cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
			myfile << " " << cohpit->getb();
			cohpit++;
			nc++;
		}
		myfile << endl;
		
		//adjust stepsize (stepsize/2) because one of the predicted values is <=0? if one of the predicted values is <=0: set (int) iadj=1
		iadj = 0;
		if((Sp<=0)||(Dp<=0))
			iadj = 1;
		else{
			for(int k=0;((k<bp.length())&&(iadj!=1));++k){
				if(bp(k)<=0)
					iadj = 1;
			}
		}
		
		while(iadj==1){
			myfile << "one of the predicted values is <=0, this is not allowed, so adjust stepsize (stepsize/2)" << endl;
			if(step==step_m){
				myfile << "stepsize already has reached minimum value " << step_m << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			step = RCONST(step/2.0);
			if(step<step_m){
				step = step_m;
				myfile << "minimal step reached" << endl;
			}
			myfile << "new stepsize = " << step << endl;
			
			Sp = RCONST(Sa+step*vn(0));
			Dp = RCONST(Da+step*vn(vn.length()-1));
			myfile << "predicted values:" << endl;
			myfile << Sp << " " << Dp;
			cohp.assign(coh.begin(),coh.end());
			cohpit = cohp.begin();
			int nc = 1; //number of cohort that is changed (nc=1,...,N) => correction vn(nc) => adjusted b in bp(nc-1)
			bp.setlength(coh.size());
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				bp(nc-1) = RCONST(cohit->getb()+step*vn(nc));
				cohpit->set_init(cohit->getMass(),cohit->getX(),cohit->getY(),cohit->getA(),bp(nc-1));
				myfile << " " << cohpit->getb();
				cohpit++;
				nc++;
			}
			myfile << endl;
			iadj = 0;
			if((Sp<=0)||(Dp<=0))
				iadj = 1;
			else{
				for(int k=0;((k<bp.length())&&(iadj!=1));++k){
					if(bp(k)<=0)
						iadj = 1;
				}
			}
		}

		//adjust mesh points (change of cohp and Sp)
		myfile << "ADJUST MESH OF PREDICTION" << endl;
		S = Sp;
		D = Dp;
		integrate_allcohorts(cohp,out,0);
		if(tempS<=0)
			Sp = 0.000001;
		else
			Sp = tempS;
		cohp = cohnew;
		myfile << "adjusted Sp: " << Sp << endl;
		myfile << "adjusted number of cohorts: " << cohp.size() << endl;
		myfile << "adjusted mesh points:" << endl;
		for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit)
			myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
		myfile << endl;

		//Newton-correction 

		// first step
		j = 1;

		//calculation of Jacobian J2
		J1 = JacobianGD(Sp,Dp,cohp,out);
		myfile << " Jacobian J1:" << endl;
		for(int k=0;k!=J1.rows();++k){
			for(int l=0;l!=J1.cols();++l)
				myfile << J1(k,l) << " ";
			myfile << endl;
		}
		myfile << endl;

		//adjust J2 to calculate vp
		J2.setlength(J1.cols(),J1.cols());
		rl.setlength(J1.cols());
		for(int k=0;k<J1.rows();++k){
			for(int l=0;l<J1.cols();++l)
				J2(k,l) = J1(k,l);
			rl(k) = 0;
		}
		rl(J1.rows())=1;
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = 1;
		alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
		myfile << " vp (unstandardised)=" << endl;
		for(int k=0;k<J1.cols();++k)
			myfile << v(k) << " ";
		myfile << endl;
		vp = normalize(v);
		myfile << "vp (normalized)=" << endl;
		for(int k=0;k<vp.length();++k)
			myfile << vp(k) << " ";
		myfile << endl;

		//adjust J2 to (J1 vp)^T
		for(int l=0;l<J1.cols();++l)
			J2(J1.rows(),l) = vp(l);

		//calculation of right-hand side R = predicted values - values after 1 map iteration 
		S = Sp;
		D = Dp;
		mapM_1(cohp,out);
		R.setlength(J2.cols());
		if(tempS<=0){
			myfile << "values after 1 map iteration (S,b): " << 0.000001;
			R(0) = RCONST(Sp-0.000001);
		}
		else{
			myfile << "values after 1 map iteration (S,b): " << tempS;
			R(0) = RCONST(Sp-tempS);
		}

		int it = 1;
		cohpit = cohp.begin();
		for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
			myfile << " " << cohit->getb(); 
			R(it) = RCONST(cohpit->getb()-cohit->getb());
			it++;
			cohpit++;
		}
		myfile << endl;
		R(J2.cols()-1) = RCONST(0);
		myfile << "values right-hand side R =";
		normR = RCONST(0);
		for(int l=0;l<R.length();++l){
			myfile << " " << R(l);
			normR += abs(R(l));
		}
		myfile << endl;

		//solve equation J2*cor=R => cor
		alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
		myfile << "values correction = ";
		normcor = RCONST(0);
		for(int l=0;l<cor.length();++l){
			myfile << " " << cor(l);
			normcor += abs(cor(l));
		}
		myfile << endl;
		myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance<< ")" << endl;
		myfile << "number of Newton steps taken = " << j << endl;

		// next possible steps

		while((j<=Newtonmax)&&((normcor>VarTolerance)||(normR>FunTolerance))){

			//check if R & cor are small enough (if not: the Newton correction step has to be repeated) (if number of Newton steps taken <= Newtonmax, if more steps are already taken: adjust stepsize and retake prediction)
			myfile << "Newton correction step has to be repeated " << endl;
			// adjust prediction:
			Sp += demping*cor(0);
			Dp += demping*cor(cor.length()-1);
			myfile << "adjusted prediction = " << Sp;
			it = 1;
			for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
				cohpit->setb(RCONST(cohpit->getb()+demping*cor(it)));
				it++;
				myfile << " " << cohpit->getb();
			}
			myfile << " " << Dp << endl;

			//adjust mesh points (change of cohp and Sp)
			myfile << "ADJUST MESH OF PREDICTION" << endl;
			S = Sp;
			D = Dp;
			integrate_allcohorts(cohp,out,0);
			if(tempS<=0)
				Sp = 0.000001;
			else
				Sp = tempS;
			cohp = cohnew;
			myfile << "adjusted Sp: " << Sp << endl;
			myfile << "adjusted number of cohorts: " << cohp.size() << endl;
			myfile << "adjusted mesh points:" << endl;
			for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
				myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() <<endl;
			}
			myfile << endl;

			myfile << "Da = " << Da << ", Sa = " << Sa << endl;
			myfile << endl;

			//Newton-correction 

			//calculation of Jacobian J2
			J1 = JacobianGD(Sp,Dp,cohp,out);
			myfile << " Jacobian J1:" << endl;
			for(int k=0;k!=J1.rows();++k){
				for(int l=0;l!=J1.cols();++l)
					myfile << J1(k,l) << " ";
				myfile << endl;
			}
			myfile << endl;

			//adjust J2 to calculate vp
			J2.setlength(J1.cols(),J1.cols());
			rl.setlength(J1.cols());
			for(int k=0;k<J1.rows();++k){
				for(int l=0;l<J1.cols();++l)
					J2(k,l) = J1(k,l);
				rl(k) = 0;
			}
			rl(J1.rows()) = 1;
			for(int l=0;l<J1.cols();++l)
				J2(J1.rows(),l) = 1;
			alglib::rmatrixsolve(J2,J1.cols(),rl,info,rep,v);
			myfile << " vp (unstandardised)=" << endl;
			for(int k=0;k<J1.cols();++k)
				myfile << v(k) << " ";
			myfile << endl;
			vp = normalize(v);
			myfile << "vp (normalized)=" << endl;
			for(int k=0;k<vp.length();++k)
				myfile << vp(k) << " ";
			myfile << endl;

			//adjust J2 to (J1 vp)^T
			for(int l=0;l<J1.cols();++l)
				J2(J1.rows(),l) = vp(l);

			//calculation of right-hand side R = predicted values - values after 1 map iteration 
			S = Sp;
			D = Dp;
			mapM_1(cohp,out);
			R.setlength(J2.cols());
			if(tempS<=0){
				myfile << "values after 1 map iteration (S,b): " << 0.000001;
				R(0) = RCONST(Sp-0.000001);
			}
			else{
				myfile << "values after 1 map iteration (S,b): " << tempS;
				R(0) = RCONST(Sp-tempS);
			}

			it = 1;
			cohpit = cohp.begin();
			for(cohit=cohadj.begin();cohit!=cohadj.end();cohit++){
				myfile << " " << cohit->getb(); 
				R(it) = RCONST(cohpit->getb()-cohit->getb());
				it++;
				cohpit++;
			}
			myfile << endl;
			R(J2.cols()-1) = RCONST(0);
			myfile << "values right-hand side R =";
			normR = RCONST(0);
			for(int l=0;l<R.length();++l){
				myfile << " " << R(l);
				normR += abs(R(l));
			}
			myfile << endl;

			//solve equation J2*cor=R => cor
			alglib::rmatrixsolve(J2,J1.cols(),R,info,rep,cor);
			myfile << "values correction = ";
			normcor = RCONST(0);
			for(int l=0;l<cor.length();++l){
				myfile << " " <<cor(l);
				normcor += abs(cor(l));
			}
			myfile << endl;
			j++;
			myfile << "norm correction = " << normcor << " (compare to " << VarTolerance << "), norm R = " << normR << " (compare to " << FunTolerance<< ")" << endl;
			myfile << "number of Newton steps taken = " << j << endl;
		}

		
		if((normcor<=VarTolerance)&&(normR<=FunTolerance)){
			//IF SMALL ENOUGH: new fixed point = predicted + correction (with adjusted mesh points)
			myfile << "correction small enough " << endl;

			// adjust prediction:
			Sp += demping*cor(0);
			Dp += demping*cor(cor.length()-1);
			myfile << "adjusted prediction = " << Sp;
			int it = 1;
			for(cohpit=cohp.begin();cohpit!=cohp.end();cohpit++){
				cohpit->setb(RCONST(cohpit->getb()+demping*cor(it)));
				it++;
				myfile << " " << cohpit->getb();
			}
			myfile << " " << Dp << endl;

			//adjust mesh points (change of cohp and Sp)
			myfile << "ADJUST MESH OF PREDICTION" << endl;
			S = Sp;
			D = Dp;
			integrate_allcohorts(cohp,out,0);
			if(tempS<=0)
				Sp = 0.000001;
			else
				Sp = tempS;
			cohp = cohnew;
			myfile << "adjusted Sp (of new fixed point): " << Sp << endl;
			myfile << "adjusted number of cohorts (of new fixed point): " << cohp.size() << endl;
			myfile << "adjusted mesh points (of new fixed point):" << endl;
			for(cohpit=cohp.begin();cohpit!=cohp.end();++cohpit){ 
				myfile << cohpit->getMass() << " " << cohpit->getX() << " " << cohpit->getY() << " " << cohpit->getA() << " " << cohpit->getb() << endl;
			}
			myfile << endl;

			Sa = Sp;
			Da = Dp;
			coh.assign(cohp.begin(),cohp.end());
			p++; //number of calculated fixed points + 1
			myfile << "RESULTS continuation: " << endl;
			myfile << p << " " << Sa << " " << Da << " " << endl;
			myfile1<< p << " " << Sa << " " << Da << " " << endl;
			myfile2 << Da << " " << coh.size() << " " << Sa << " "; 
			for(cohit=coh.begin();cohit!=coh.end();++cohit){
				myfile << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
				myfile1 << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << endl;
				btot += cohit->getb();
			}
			myfile << endl;
			myfile1 << endl;
			myfile2 << btot << endl;
			btot = 0;

			//adjust stepsize according to number of Newton steps taken
			if(j>=4){
				step = RCONST(step/2.0); 
				myfile << "more than 4 or 4 Newton steps needed, so stepsize divided by 2" << endl;
				if(step<step_m){
					step = step_m;
					myfile << "minimal Newton step reached" << endl;
				}
				myfile << "new stepsize = " << step << endl;
				myfile << endl;
				myfile << endl;
			}
			else{
				step = RCONST(step*1.3);
				myfile << "less than 4 Newton steps needed, so stepsize * 1.3" << endl;
				if(step>step_M){
					step = step_M;
					myfile << "maximal Newton step reached" << endl;
				}
				myfile << "new stepsize = " << step << endl;
				myfile << endl;
				myfile << endl;
			}
		}

		else{ //more than Newtonmax steps taken, so adjust stepsize and retake prediction
			if(step==step_m){
				myfile << "more than " << Newtonmax << " Newton steps needed and stepsize has reached minimum value " << step_m << endl;
				myfile << "end continuation" << endl;
				return 1;
			}
			step = RCONST(step/2.0); 
			if(step<step_m){
				step = step_m;
				myfile << "minimal Newton step reached" << endl;
			}
			myfile << "more than " << Newtonmax << " Newton steps needed, so stepsize/2 for prediction and tried again" << endl;
			myfile << endl;
			q = 1;
		}
	}
	myfile.close();
	return 0;
}

int main() {
	myfile_OC.precision(10);
	myfile_OC.open("outputcohorts.txt", ios::app);
	clist coh;
	cohort a;
	realtype delta_m;
	int N = 100;
	if(cond_min==1)
	  delta_m = RCONST(((1-phi)*m_max-phi*m_min)/N);
	else
	  delta_m = RCONST(((1-phi)*m_max)/N);
	for(int i=1; i<=N; i++){
	  if(cond_min==1)
		a.set_init(phi*m_min+(i-0.5)*delta_m,0.1,0.5,0.5,0.0000001);
	  else
		a.set_init((i-0.5)*delta_m,0.1,0.5,0.5,0.0000001);
	  coh.push_back(a);
	}
	loopMap(1.0,coh,300,1,0);
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

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data){ //ODEs for m,X,Y,A,F,theta
	realtype m, X, Y, A, F, theta;
	m = NV_Ith_S(y,0);
	X = NV_Ith_S(y,1);
	Y = NV_Ith_S(y,2);
	A = NV_Ith_S(y,3);
	F = NV_Ith_S(y,NDIM);
	theta = NV_Ith_S(y,NDIM+1);

	if(cond_max==0)
		NV_Ith_S(ydot,0) = RCONST(mu)*m*(1-m/RCONST(m_max))*RCONST(c1)*S/(RCONST(zeta1)+S);
	if(cond_max==1)
		NV_Ith_S(ydot,0) = RCONST(mu)*m*RCONST(c1)*S/(RCONST(zeta1)+S);
	NV_Ith_S(ydot,1) = RCONST(k1) - (RCONST(k2p)+RCONST(k2pp)*Y)*X;
	NV_Ith_S(ydot,2) = (RCONST(k3p)+RCONST(k3pp)*A)*(1-Y)/(RCONST(J3)+1-Y) - RCONST(k4)*m*X*Y/(RCONST(J4)+Y);
	NV_Ith_S(ydot,3) = RCONST(k5p)+(RCONST(k5pp)*pow(m*X,n))/(pow(RCONST(J5),n)+pow(m*X,n))-RCONST(k6)*A;
	NV_Ith_S(ydot,NDIM) = -(nu(y)+beta(y))*F;
	if(cond_max==0)
		NV_Ith_S(ydot,NDIM+1) = RCONST(c2)*RCONST(mu)*m*(1-m/RCONST(m_max))*RCONST(c1)*S/(RCONST(zeta1)+S);
	if(cond_max==1)
		NV_Ith_S(ydot,NDIM+1) = RCONST(c2)*RCONST(mu)*m*RCONST(c1)*S/(RCONST(zeta1)+S);
	return(0);
}

static int g(realtype t, N_Vector y, realtype *gout, void *user_data){ //for RootFinding during age integration
  gout[0] = NV_Ith_S(y,1)-X_div;
  return(0);
}

void cohort::PrintOutput(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6){
  /*prints the data values (t,m,X,Y,A,F,theta) of a cohort at a certain age t in the file outputcohorts.txt*/
  myfile_OC << "At t = " << t << " m = " << y1 << " X = " << y2 << " Y = " << y3 << " A = " << y4 << " F = " << y5 << " theta = " << y6 << endl;
  return;
}

void cohort::PrintOutputData(realtype t, realtype y1, realtype y2, realtype y3, realtype y4, realtype y5, realtype y6){
/*prints the data values (t,m,X,Y,A,F,theta) of a cohort at a certain age t in the file values.txt*/
  ofstream myfile;
  myfile.open ("values.txt", ios::app);
  myfile.precision(10);
  myfile << t << " " << y1 << " " << y2 << " " << y3 << " " << y4 << " " << y5 << " " << y6 << endl;
  myfile.close();
  return;
}

void printListCoh_OC(clist &coh){ /*prints the given list of cohorts in outputcohorts.txt*/
  clistit cohit;
  for(cohit=coh.begin();cohit!=coh.end();++cohit){
	myfile_OC << cohit->getMass() << " " << cohit->getX() << " " << cohit->getY() << " " << cohit->getA() << " " << cohit->getb() << " "<<endl;
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