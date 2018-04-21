#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <fftw3.h>
#include "merssene_twister.h"
/*----------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------*/
#define FRANDOM ((double) rand()/RAND_MAX)
//#define FRANDOM  genrand64_real1() /* chamando o gerador de números aleatórios*/
/*----------------------------------------------------------------------------------------*/

#define FILEOUT "teste5.dat"                        // NOMEAND0 ARQUIVO

#define PI M_PI 

#define tmax (2*3225806)                            // TEMPO MAXIMO SIMULACAO
#define nhist 100                                     // NUMERO DE HISTORIAS
#define corte 1000                                 // CORTE PARA SALVA MENOS DADOS 

#define diametro 22.80e-9                           // DIMAETRO DA PARTICULA [m]
#define dx diametro                                 // DISTANCIA ENTRE PARTICULAS [m]
#define vpart (4.0*PI*pow((diametro/2.0),3)/3.0)    // VOLUME DA PARTICULA [m³]
#define Lx 2                                        // NUMERO DE PARTICULAS 

#define Ms 5.8e5                                    // MAGNETIZACAO DE SATURACAO [A/m] 
#define gama 2.2128e5                               // RAZAO GIROMAGNETICA [m/A/s]
#define mu 1.256637061e-6                           // PERMEABILIDADE MAGNETICA [mT/A]
#define ku 3.2e4                                    // CONSTANTE DE ANISOTROPIA UNIAXIAL [J/m³]
#define kc (1.5*ku)                                     // CONSTANTE DE ANISOTROPIA CUBICA [J/m³]

#define kb 1.38064852e-23                           // CONSTANTE DE BOLTZMANN [J/K]

#define direction 0             // CONFIGURACAO INICIAL DOS MOMENTOS: 0 = ALINHADOS, 1 = ALEATORIOS
#define e_facil_u 1             // CONFIGURACAO DOS EIXOS DE ANISOTROPIA UNIAXIAL: 0 = ALINHADOS, 1 = ALEATORIOS
#define e_anis_c 0              // CONFIGURACAO DOS EIXOS DE ANISOTROPIA CUBICA: 0 = ALINHADOS, 1 = ALEATORIOS 
#define auni 0                  // CONSTANTENTES DE ANISOTROPIA: 0 = Ki=K , 1 = Ki = LOGNORMAL
#define dvar 0                  // CONFIGURACAO DAS DISTANCIAS ENTRE AS PARTICULAS: 0 = IGUALMENTE ESPACADAS   

#define histerese 1             // EXPERIMENTO DE HISTERESE

#define dt 0.01                 // PASSO DE TEMPO [s]
#define gdip (vpart/4.0/PI)     // FATOR QUE MULTIPLICA A INTERACAO DIPOLAR [m³]

double T = 25.0;                // TEMPERATURA [K]
double cp = 0.5;                // CAMPO MAXIMO/MINIMO PARA HISTERESE

#define alfa 0.01               // DAMPING 

#define dipolar 0               // LIGA/DESL INTERACAO DIPOLAR 
#define externo 1               // LIGA/DESL CAMPO EXTERNO
#define anis_u 1                // LIGA/DESL ANISOTROPIA UNIAXIAL
#define anis_c 1                // LIGA/DESL ANISOTROPIA CUBICA

int animacao = 0;               // FAZ A ANIMACAO
double plot = 0;                // ESTILO ANIMACAO 0 = 2D, 1 = 3D COM ESQUEMA DE CORES, 2 = 3D SEM ESQUEMA DE CORES

/*----------------------------------------------------------------------------------------*/
#include "f-idteste.h" 
/*----------------------------------------------------------------------------------------*/
clock_t start, stop;
/*----------------------------------------------------------------------------------------*/  
double *Jdip_xx,*Jdip_xy,*Jdip_xz,*Jdip_yy,*Jdip_yz,*Jdip_zz;
fftw_complex *Jk_xx,*Jk_xy,*Jk_xz,*Jk_yy,*Jk_yz,*Jk_zz;
fftw_plan Jxx_r2c,Jxy_r2c,Jxz_r2c,Jyy_r2c,Jyz_r2c,Jzz_r2c;

double *mx,*my,*mz, *hx,*hy,*hz;
fftw_complex *mx_k,*my_k,*mz_k,*hx_k,*hy_k,*hz_k;
fftw_plan mx_r2c,hx_c2r,my_r2c,hy_c2r,mz_r2c,hz_c2r;

void aloca_fftw(){
  mx = fftw_malloc(sizeof(double)*Lx);
  hx = fftw_malloc(sizeof(double)*Lx);
  mx_k = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  hx_k = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  mx_r2c = fftw_plan_dft_r2c_1d(Lx,mx,mx_k,FFTW_PATIENT);
  hx_c2r = fftw_plan_dft_c2r_1d(Lx,hx_k,hx,FFTW_PATIENT);

  my = fftw_malloc(sizeof(double)*Lx);
  hy = fftw_malloc(sizeof(double)*Lx);
  my_k = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  hy_k = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  my_r2c = fftw_plan_dft_r2c_1d(Lx,my,my_k,FFTW_PATIENT);
  hy_c2r = fftw_plan_dft_c2r_1d(Lx,hy_k,hy,FFTW_PATIENT);
  
  mz = fftw_malloc(sizeof(double)*Lx);
  hz = fftw_malloc(sizeof(double)*Lx);
  mz_k = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  hz_k = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  mz_r2c = fftw_plan_dft_r2c_1d(Lx,mz,mz_k,FFTW_PATIENT);
  hz_c2r = fftw_plan_dft_c2r_1d(Lx,hz_k,hz,FFTW_PATIENT);

  Jdip_xx = fftw_malloc(sizeof(double)*Lx);
  Jk_xx = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  Jxx_r2c = fftw_plan_dft_r2c_1d(Lx,Jdip_xx,Jk_xx,FFTW_PATIENT);

  Jdip_xy = fftw_malloc(sizeof(double)*Lx);
  Jk_xy = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  Jxy_r2c = fftw_plan_dft_r2c_1d(Lx,Jdip_xy,Jk_xy,FFTW_PATIENT);

  Jdip_xz = fftw_malloc(sizeof(double)*Lx);
  Jk_xz = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  Jxz_r2c = fftw_plan_dft_r2c_1d(Lx,Jdip_xz,Jk_xz,FFTW_PATIENT);

  Jdip_yy = fftw_malloc(sizeof(double)*Lx);
  Jk_yy = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  Jyy_r2c = fftw_plan_dft_r2c_1d(Lx,Jdip_yy,Jk_yy,FFTW_PATIENT);

  Jdip_yz = fftw_malloc(sizeof(double)*Lx);
  Jk_yz = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  Jyz_r2c = fftw_plan_dft_r2c_1d(Lx,Jdip_yz,Jk_yz,FFTW_PATIENT);

  Jdip_zz = fftw_malloc(sizeof(double)*Lx);
  Jk_zz = fftw_malloc(sizeof(fftw_complex)*(Lx/2+1));
  Jzz_r2c = fftw_plan_dft_r2c_1d(Lx,Jdip_zz,Jk_zz,FFTW_PATIENT);
}

void dipolar_fft(double **Jdip, double ***RR){
  int i=0, ii=0;
  for(i=0;i<Lx;i++)
    Jdip_xx[i]=Jdip_yy[i]=Jdip_zz[i]=Jdip_xy[i]=Jdip_xz[i]=Jdip_yz[i]=0;
  
  for(i=0;i<Lx;i++){
    Jdip_xx[i] =  Jdip[0][i]*(3*pow(RR[0][0][i],2)-1)/Lx;
    Jdip_yy[i] = -Jdip[0][i]/Lx;
    Jdip_zz[i] = -Jdip[0][i]/Lx;
    
    Jdip_xy[i] = 0;
    Jdip_xz[i] = 0;
    Jdip_yz[i] = 0;
    
  }
  
  fftw_execute(Jxx_r2c);
  fftw_execute(Jyy_r2c);
  fftw_execute(Jzz_r2c);
  
  fftw_execute(Jxy_r2c);
  fftw_execute(Jxz_r2c);
  fftw_execute(Jyz_r2c);
}

void campo_demag(double *mmx, double *mmy, double *mmz){
  int i,j,k,c;
  
  for(i=0;i<Lx;i++){
    mx[i] = mmx[i]; 
    my[i] = mmy[i]; 
    mz[i] = mmz[i]; 
  }
  
  fftw_execute(mx_r2c);
  fftw_execute(my_r2c);
  fftw_execute(mz_r2c);
  
  for(i=0;i<Lx/2+1;i++){
    for(c=0;c<2;c++){
      hx_k[i][c] = Jk_xx[i][0]*mx_k[i][c];
      hy_k[i][c] = Jk_yy[i][0]*my_k[i][c];
      hz_k[i][c] = Jk_zz[i][0]*mz_k[i][c];
    }
  }
  
  fftw_execute(hx_c2r);
  fftw_execute(hy_c2r);
  fftw_execute(hz_c2r); 
}
double distmin(double *rrr, int lx, double ***RR, double **Jdip){
  
  int i,j,k,ii,jj,kk,di,dj,dk;
  double Distimin, Dist;
  
  for(i=0; i<lx; i++){
    for(ii=0;ii<lx;ii++){
      if(i==ii)
	Jdip[i][ii]=0;
      else{
	int di;
	double Distmin = sqrt(pow(rrr[ii]- rrr[i],2)); 
	double Rmin[3] = {rrr[ii] - rrr[i],0,0}; 
	for(di=-1;di<=1;di++){
	  double des = di*Lx*dx;
	  Dist = sqrt(pow(rrr[ii] + des*0 - rrr[i],2));
	  if(Dist<Distmin){
	    Distmin=Dist;
	    Rmin[0]=rrr[ii]+des*0 - rrr[i];
	    Rmin[1]=0;
	    Rmin[2]=0;
	  }}				  
	Dist = Distmin;
	double MOD=sqrt(pow(Rmin[0],2));
	Jdip[i][ii]  = gdip/pow(Dist,3);
	RR[0][i][ii] = Rmin[0]/MOD;
	RR[1][i][ii] = 0.0;
	RR[2][i][ii] = 0.0;
      }}}
}
/*******************************************************************************************/
int main(){
  
  int i,ii,hc,nh,j=0;
  long int t;
  long double tempo = 0;
  double mmp,mmm,ku_medio=0,kc_medio=0;
  double mxm = 0, mym = 0, mzm = 0;
  double *Mx = {0}, *My = {0}, *Mz = {0};
  double *mxmt = {0}, *mymt = {0}, *mzmt = {0};
  double *mxmmt = {0}, *mymmt = {0}, *mzmmt = {0};
  double *eax_x= {0}, *eax_y= {0}, *eax_z= {0};
  double *eax_xc= {0}, *eax_yc= {0}, *eax_zc= {0};
  double *dw_x = {0}, *dw_y = {0}, *dw_z = {0};
  double *Hz_x = {0}, *Hz_y = {0}, *Hz_z = {0};
  double *Hau_x = {0}, *Hau_y = {0}, *Hau_z = {0};
  double *Hac_x = {0}, *Hac_y = {0}, *Hac_z = {0};
  double *Hd_x = {0}, *Hd_y = {0}, *Hd_z = {0};
  double *Hef_x = {0}, *Hef_y = {0}, *Hef_z = {0};
  double *Jdip_x = {0};//, **Jdip_y = {0}, **Jdip_z = {0};
  double **Jdip = {0};//, **Jdip_y = {0}, **Jdip_z = {0};
  double *rr_x = {0}, *rr_y = {0}, *rr_z = {0};
  double *Rx = {0};//, **Ry = {0}, **Rz = {0};
  double ***R = {0};//, **Ry = {0}, **Rz = {0};
  double *Ku= {0},*Kc= {0}, Dist=0, *hex_x={0},*hex_y={0},*hex_z={0};
  double cext = 0, *campo={0},*timex={0};
  double *mhx, *mhy, *mhz, *mmhx,*mmhy,*mmhz;
  double *Mhx, *Mhy, *Mhz, *Mmhx,*Mmhy,*Mmhz;
  double *ffx, *fftx, *ggx,*ggtx, *magtx, *magnewx;
  double *ffy, *ffty, *ggy,*ggty, *magty, *magnewy;
  double *ffz, *fftz, *ggz,*ggtz, *magtz, *magnewz;
  
  Rx =    (double*)calloc(Lx,sizeof(double)*Lx); Jdip_x= (double *)calloc(Lx,sizeof(double)*Lx);
  Mx =    (double*)calloc(Lx,sizeof(double)*Lx); My=     (double*)calloc(Lx,sizeof(double)*Lx); Mz=     (double*)calloc(Lx,sizeof(double)*Lx);
  Hz_x =  (double*)calloc(Lx,sizeof(double)*Lx); Hz_y=   (double*)calloc(Lx,sizeof(double)*Lx); Hz_z=  (double*)calloc(Lx,sizeof(double)*Lx); 
  Hau_x=  (double*)calloc(Lx,sizeof(double)*Lx); Hau_y=  (double*)calloc(Lx,sizeof(double)*Lx); Hau_z=  (double*)calloc(Lx,sizeof(double)*Lx); 
  Hac_x=  (double*)calloc(Lx,sizeof(double)*Lx); Hac_y=  (double*)calloc(Lx,sizeof(double)*Lx); Hac_z=  (double*)calloc(Lx,sizeof(double)*Lx);
  Hd_x=   (double*)calloc(Lx,sizeof(double)*Lx); Hd_y=   (double*)calloc(Lx,sizeof(double)*Lx); Hd_z=   (double*)calloc(Lx,sizeof(double)*Lx); 
  Hef_x=  (double*)calloc(Lx,sizeof(double)*Lx); Hef_y=  (double*)calloc(Lx,sizeof(double)*Lx); Hef_z=  (double*)calloc(Lx,sizeof(double)*Lx); 
  eax_x=  (double*)calloc(Lx,sizeof(double)*Lx); eax_y=  (double*)calloc(Lx,sizeof(double)*Lx); eax_z=  (double*)calloc(Lx,sizeof(double)*Lx);
  eax_xc= (double*)calloc(Lx,sizeof(double)*Lx); eax_yc= (double*)calloc(Lx,sizeof(double)*Lx); eax_zc= (double*)calloc(Lx,sizeof(double)*Lx);
  dw_x=   (double*)calloc(Lx,sizeof(double)*Lx); dw_y=   (double*)calloc(Lx,sizeof(double)*Lx); dw_z=   (double*)calloc(Lx,sizeof(double)*Lx); 
  rr_x=   (double*)calloc(Lx,sizeof(double)*Lx); Ku=     (double*)calloc(Lx,sizeof(double)*Lx); Kc=     (double*)calloc(Lx,sizeof(double)*Lx);
  hex_x=  (double*)calloc(Lx,sizeof(double)*Lx); hex_y=  (double*)calloc(Lx,sizeof(double)*Lx); hex_z=  (double*)calloc(Lx,sizeof(double)*Lx);
  
  mxmt=   (double*)calloc(tmax/corte,sizeof(double)*tmax/corte); mymt=  (double*)calloc(tmax/corte,sizeof(double)*tmax/corte);
  campo=  (double*)calloc(tmax/corte,sizeof(double)*tmax/corte); timex= (double*)calloc(tmax/corte,sizeof(double)*tmax/corte);
  mzmt=   (double*)calloc(tmax/corte,sizeof(double)*tmax/corte); mxmmt= (double*)calloc(tmax/corte,sizeof(double)*tmax/corte);
  mymmt=  (double*)calloc(tmax/corte,sizeof(double)*tmax/corte); mzmmt= (double*)calloc(tmax/corte,sizeof(double)*tmax/corte);
  
  R = (double ***)calloc(3,sizeof(double**)*3);
  for(hc = 0; hc < 3; hc++ ){
    R[hc] = (double **)calloc(Lx,sizeof(double*)*Lx);
    for(i=0;i<Lx;i++)
      R[hc][i] = (double *)calloc(Lx,sizeof(double)*Lx);
  }  
  Jdip = (double **)calloc(Lx,sizeof(double*)*Lx);
  for(i=0;i<Lx;i++)
    Jdip[i] = (double *)calloc(Lx,sizeof(double)*Lx);
  
  mhx    = (double*)calloc(Lx,sizeof(double)*Lx);
  mhy    = (double*)calloc(Lx,sizeof(double)*Lx);
  mhz    = (double*)calloc(Lx,sizeof(double)*Lx);
  mmhx   = (double*)calloc(Lx,sizeof(double)*Lx);
  mmhy   = (double*)calloc(Lx,sizeof(double)*Lx);
  mmhz   = (double*)calloc(Lx,sizeof(double)*Lx);
  
  Mhx    = (double*)calloc(Lx,sizeof(double)*Lx);
  Mhy    = (double*)calloc(Lx,sizeof(double)*Lx);
  Mhz    = (double*)calloc(Lx,sizeof(double)*Lx);
  
  Mmhx   = (double*)calloc(Lx,sizeof(double)*Lx);
  Mmhy   = (double*)calloc(Lx,sizeof(double)*Lx);
  Mmhz   = (double*)calloc(Lx,sizeof(double)*Lx);

  ffx       = (double*)calloc(Lx,sizeof(double)*Lx);
  fftx      = (double*)calloc(Lx,sizeof(double)*Lx);
  ggx       = (double*)calloc(Lx,sizeof(double)*Lx);
  ggtx      = (double*)calloc(Lx,sizeof(double)*Lx);
  magtx     = (double*)calloc(Lx,sizeof(double)*Lx);
  magnewx   = (double*)calloc(Lx,sizeof(double)*Lx);
  
  ffy       = (double*)calloc(Lx,sizeof(double)*Lx);
  ffty      = (double*)calloc(Lx,sizeof(double)*Lx);
  ggy       = (double*)calloc(Lx,sizeof(double)*Lx);
  ggty      = (double*)calloc(Lx,sizeof(double)*Lx);
  magty     = (double*)calloc(Lx,sizeof(double)*Lx);
  magnewy   = (double*)calloc(Lx,sizeof(double)*Lx);
  
  ffz       = (double*)calloc(Lx,sizeof(double)*Lx);
  fftz      = (double*)calloc(Lx,sizeof(double)*Lx);
  ggz       = (double*)calloc(Lx,sizeof(double)*Lx);
  ggtz      = (double*)calloc(Lx,sizeof(double)*Lx);
  magtz     = (double*)calloc(Lx,sizeof(double)*Lx);
  magnewz   = (double*)calloc(Lx,sizeof(double)*Lx);
/*----------------------------------------------------------------------------------------*/    
/*####################################################################################################*/
  unsigned long long init[4] = {0x87654ULL, 0x76543ULL, 0x65432ULL, 0x54321ULL}, length = 4; //Seed.
  init_by_array64(init, length);
  //genrand64_real1() => [0, 1]. //genrand64_real2() => [0, 1). //genrand64_real3() => (0, 1).
/*####################################################################################################*/
/*----------------------------------------------------------------------------------------*/
  FILE *f1;
  f1 = fopen(FILEOUT,"w");
/*----------------------------------------------------------------------------------------*/
  for(nh=0;nh<nhist;nh++){ //número de histórias
/*----------------------------------------------------------------------------------------*/
    ordem_momentos(direction,Lx,Mx,My,Mz);
/*----------------------------------------------------------------------------------------*/
    ordem_eixos_anis_uni(e_facil_u,eax_x,eax_y,eax_z, Lx);
/*----------------------------------------------------------------------------------------*/
    ordem_eixos_anis_cub(e_anis_c,eax_xc,eax_yc,eax_zc, Lx);
/*----------------------------------------------------------------------------------------*/
    ordem_const_anis_uni(anis_u,anis_c,auni,Lx,ku,kc,Ku,Kc);
/*----------------------------------------------------------------------------------------*/
    ordem_posicao(dvar,Lx,dx,rr_x);    
/*----------------------------------------------------------------------------------------*/
    if (externo == 1){
      for(i=0; i<Lx; i++){
        hex_x[i] = 1.0;
        hex_y[i] = 0.0;
        hex_z[i] = 0.0;
        
        double mod = sqrt(pow(hex_x[i],2) + pow(hex_y[i],2) + pow(hex_z[i],2));
        
        hex_x[i] = hex_x[i]/mod; 
        hex_y[i] = hex_y[i]/mod;
        hex_z[i] = hex_z[i]/mod;
      }
    }
    else{
      for(i=0; i<Lx; i++){
        hex_x[i] = 0;
	hex_y[i] = 0;
	hex_z[i] = 0;
      }
    }
/*----------------------------------------------------------------------------------------*/
    if(dipolar == 1){        
      distmin(rr_x,Lx,R,Jdip);
      aloca_fftw();
      dipolar_fft(Jdip,R);
      campo_demag(Mx,My,Mz);
    }
/*----------------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------------------------------------------------------------------*/ 
    for(t=0 ; t<tmax; t++){            // TEMPO DIMULACAO  
/*------------------------------------------------------------------------------------------------------------------------------------------*/ 
/*----------------------------------------------------------------------------------------*/
      zerar_campos(Lx,Hef_x,Hef_y,Hef_z, Hz_x,Hz_y,Hz_z,Hd_x,Hd_y,Hd_z,Hau_x,Hau_y,Hau_z,Hac_x,Hac_y,Hac_z);

      mxm = 0; mym = 0; mzm = 0;
/*-----------------------------------------------------------------------------------------*/  
      if(dipolar == 1)
	campo_demag(Mx,My,Mz);
/*----------------------------------------------------------------------------------------*/        
/*------------------------------------------------------------------------------------------------------------------------------------------*/
      for (i=0;i<Lx;i++){              // VARIACAO DAS PARTICULAS NA REDE
/*------------------------------------------------------------------------------------------------------------------------------------------*/
        if (T != 0 || alfa != 0 ){     // CAMPO ESTOCÁSTICO
	  dw_x[i] = ngaussian()*sqrt(2*alfa*kb*T*dt/Ms/Ms/mu/vpart);
	  dw_y[i] = ngaussian()*sqrt(2*alfa*kb*T*dt/Ms/Ms/mu/vpart);
	  dw_z[i] = ngaussian()*sqrt(2*alfa*kb*T*dt/Ms/Ms/mu/vpart);
        }
        else{
	  dw_x[i] = 0;
	  dw_y[i] = 0;
	  dw_z[i] = 0;
	}
/*----------------------------------------------------------------------------------------*/
        if (anis_u == 1){              // ANISOTROPIA UNIAXIAL
	  Hau_x[i] = (2.0*Ku[i]/mu/Ms/Ms)*Mx[i]*eax_x[i];
	  Hau_y[i] = (2.0*Ku[i]/mu/Ms/Ms)*My[i]*eax_y[i];
	  Hau_z[i] = (2.0*Ku[i]/mu/Ms/Ms)*Mz[i]*eax_z[i];
        }
        else{ 
	  Hau_x[i] = 0;
	  Hau_y[i] = 0;
	  Hau_z[i] = 0;
        }
/*----------------------------------------------------------------------------------------*/
        if (anis_c == 1){              // ANISOTROPIA CUBICA
	  Hac_x[i] = (2.0*Kc[i]/mu/Ms/Ms)*(pow(Mx[i],3)*eax_xc[i]);
	  Hac_y[i] = (2.0*Kc[i]/mu/Ms/Ms)*(pow(My[i],3)*eax_yc[i]);
	  Hac_z[i] = (2.0*Kc[i]/mu/Ms/Ms)*(pow(Mz[i],3)*eax_zc[i]);
        }
        else{
	  Hac_x[i] = 0;
	  Hac_y[i] = 0;
	  Hac_z[i] = 0;
        }
/*----------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------*/
        if (externo == 1){             // CAMPO EXTERNO
	  Hz_x[i] = cp*hex_x[i];
	  Hz_y[i] = cp*hex_y[i];
	  Hz_z[i] = cp*hex_z[i];
        }
        else{
	  Hz_x[i] = 0;
	  Hz_y[i] = 0;
	  Hz_z[i] = 0;
        }
/*----------------------------------------------------------------------------------------*/        
        if(dipolar == 1){               // CAMPO DIPOLAR
	  Hd_x[i] = hx[i]; 
	  Hd_y[i] = hy[i]; 
	  Hd_z[i] = hz[i];   
        }
	else
	  {
	    Hd_x[i] = 0; 
	    Hd_y[i] = 0; 
	    Hd_z[i] = 0;    
	  }
/*----------------------------------------------------------------------------------------*/        
	campo_efetivo(i,Hef_x,Hef_y,Hef_z, Hz_x,Hz_y,Hz_z,Hd_x,Hd_y,Hd_z, Hau_x,Hau_y,Hau_z,Hac_x,Hac_y,Hac_z);
/*----------------------------------------------------------------------------------------*/                
        tempo = t*dt;
/*----------------------------------------------------------------------------------------*/                
        mxm += Mx[i]/Lx;
        mym += My[i]/Lx;
        mzm += Mz[i]/Lx;
/*----------------------------------------------------------------------------------------*/       
                            /*NORMALIZANDO A MAGNETIZACAO*/        
        Mx[i] = Mx[i]/sqrt(pow(Mx[i],2) + pow(My[i],2) + pow(Mz[i],2));
        My[i] = My[i]/sqrt(pow(Mx[i],2) + pow(My[i],2) + pow(Mz[i],2));
        Mz[i] = Mz[i]/sqrt(pow(Mx[i],2) + pow(My[i],2) + pow(Mz[i],2));
/*----------------------------------------------------------------------------------------*/	 
/*------------------------------------------------------------------------------------------------------------------------------------------*/
      }                         // VARIACAO DAS PARTICULAS NA REDE
/*------------------------------------------------------------------------------------------------------------------------------------------*/ 
      if(t%corte==0){
        mxmt[j] += mxm;
        mymt[j] += mym;
        mzmt[j] += mzm;
	
	  campo[j] = cp;
	  timex[j] = (double) tempo;
	
	j++;
      }
      
      update(alfa,T,Hef_x,Hef_y,Hef_z,dw_x,dw_y,dw_z, Lx,mhx, mhy, mhz, mmhx,mmhy,mmhz,Mhx, Mhy, Mhz, Mmhx,Mmhy,Mmhz,ffx,fftx, ggx,ggtx, magtx, magnewx,   ffy,ffty, ggy,ggty, magty, magnewy, ffz,fftz, ggz,ggtz, magtz, magnewz,Mx,My,Mz);
      
      if(animacao == 1){ 
	anima(animacao,rr_x, rr_x, rr_x, plot,Lx,Mx,My,Mz);
      }
      if(histerese==1){    
	if(t<tmax*0.5)
	  cp = cp - 3.1e-7;
	else
	  cp = cp + 3.1e-7;
      }
      /*if(histerese!=1){    
	if(t%corte==0)
	  fprintf(f1,"%Lg %g %g %g \n",tempo,mxm,mym,mzm);
      *///}
/*------------------------------------------------------------------------------------------------------------------------------------------*/
    } // TEMPO SIMULACAO
/*------------------------------------------------------------------------------------------------------------------------------------------*/ 
    for(j=0;j<tmax/corte;j++){
      mxmmt[j] = mxmt[j]/nhist;
      mymmt[j] = mymt[j]/nhist;
      mzmmt[j] = mzmt[j]/nhist;
    }
    j=0;
/*------------------------------------------------------------------------------------------------------------------------------------------*/ 
  } // NUMERO DE HISTORIAS
/*------------------------------------------------------------------------------------------------------------------------------------------*/ 
//   if(histerese!=1){
    for(j=0;j<tmax/corte;j++){
      fprintf(f1,"%g %g %g %g %g \n",timex[j],campo[j],mxmmt[j],mymmt[j],mzmmt[j]);
//     }
  }
/*----------------------------------------------------------------------------------------*/    
   stop = clock();    
  printf("t = %ld --- Time =%f min\n",t,(float)(stop-start)/(CLOCKS_PER_SEC)/60.);
/*----------------------------------------------------------------------------------------*/  
  fclose(f1);
/*----------------------------------------------------------------------------------------*/
  free(Mx);free(My);free(Mz);free(rr_x);
  free(Hef_x);free(Hef_y);free(Hef_z);
  free(dw_x);free(dw_y);free(dw_z);
  free(Hau_x);free(Hau_y);free(Hau_z);
  free(Hz_x);free(Hz_y);free(Hz_z);
  free(Hd_x);free(Hd_y);free(Hd_z);
  free(eax_x);free(eax_y);free(eax_z);
  free(hex_x);free(hex_y);free(hex_z);
  free(Rx);free(Jdip_x);free(Ku);
  free(mhx);free(mmhx);free(mhy);
  free(mmhy);free(mhz);free(mmhz);
  free(Mhx);free(Mmhx);free(Mhy);
  free(Mmhy);free(Mhz);free(Mmhz);
  free(ffx);free(fftx);free(ggx);
  free(ggtx);free(magtx);free(magnewx);
  free(ffy);free(ffty);free(ggy);
  free(ggty);free(magty);free(magnewy);
  free(ffz);free(fftz);free(ggz);
  free(ggtz);free(magtz);free(magnewz);
/*----------------------------------------------------------------------------------------*/
} // FINAL DO MAIN

