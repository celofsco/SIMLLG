double ngaussian(void){
  static int iset=0;
  static double gset;
  double fac,r,v1,v2;
  double muu = 0.0;
  double s = 1.0;
  if (iset==0) {
    do {
      v1=2.0*FRANDOM-1.0;
      v2=2.0*FRANDOM-1.0;
      r=v1*v1+v2*v2;
    }
    while (r>=1 || r==0.0);
    fac=sqrt(-s*log(r)/r);
    gset=muu + v1*fac;
    iset=1;
    return muu + v2*fac;
  }
  else {
    iset=0;
    return gset;
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
double lognormal (void){
  double x,y,muu,sig;
  muu = 0.0;
  sig = 0.5;
  x = ngaussian();
  return y = exp(muu + sig*x);
}  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void prodvet(double *v1x, double *v1y, double *v1z, double *v2x, double *v2y, double *v2z, int lx, double *result_x, double *result_y, double *result_z)
{
  int i,j,k,hc;
  for(i=0;i<lx;i++ ){
    result_x[i] = v1y[i]*v2z[i] - v1z[i]*v2y[i];
    result_y[i] = v1z[i]*v2x[i] - v1x[i]*v2z[i];
    result_z[i] = v1x[i]*v2y[i] - v1y[i]*v2x[i];
  }
  return;
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void f(double af, double *mx,double *my,double *mz, double *hx, double *hy, double *hz, int lx, double *mhx,double *mhy,double *mhz,double *mmhx,double *mmhy,double *mmhz,double *result_x,double *result_y,double *result_z)
{
  int i;
  
  prodvet(mx,my,mz,hx,hy,hz,lx,mhx,mhy,mhz);
  prodvet(mx,my,mz,mhx,mhy,mhz,lx,mmhx,mmhy,mmhz);
  
  for(i=0;i<lx;i++){
    
    result_x[i] = -(1.0/(1.0+af*af))*(mhx[i] + af*mmhx[i]);  
    result_y[i] = -(1.0/(1.0+af*af))*(mhy[i] + af*mmhy[i]);  
    result_z[i] = -(1.0/(1.0+af*af))*(mhz[i] + af*mmhz[i]);  
  }
  
  return;
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void g(double af, double *mx,double *my,double *mz, double *hx, double *hy, double *hz, int lx,double *Mhx,double *Mhy, double*Mhz,double *Mmhx,double *Mmhy,double *Mmhz, double *result_x,double *result_y,double *result_z)
{
  int i;
  
  prodvet(mx,my,mz,hx,hy,hz,lx,Mhx,Mhy,Mhz);
  prodvet(mx,my,mz,Mhx,Mhy,Mhz,lx,Mmhx,Mmhy,Mmhz);
  
  for(i=0;i<lx;i++){
    
    result_x[i] = -(1.0/(1.0+af*af))*(Mhx[i] + af*Mmhx[i]);  
    result_y[i] = -(1.0/(1.0+af*af))*(Mhy[i] + af*Mmhy[i]);  
    result_z[i] = -(1.0/(1.0+af*af))*(Mhz[i] + af*Mmhz[i]);  
  }
  
  return;
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void update(double af, double temp, double *hx, double *hy , double *hz,double *dwx, double *dwy, double *dwz,int lx ,double *mhx,double *mhy,double *mhz,double *mmhx,double *mmhy,double *mmhz, double *Mhx,double *Mhy, double*Mhz,double *Mmhx,double *Mmhy,double *Mmhz,double *ffx,double *fftx, double *ggx,double *ggtx, double *magtx, double *magnewx,   double *ffy,double *ffty, double *ggy,double *ggty, double *magty, double *magnewy, double *ffz,double *fftz, double *ggz,double *ggtz, double *magtz, double *magnewz , double *mx, double *my, double *mz )
{
  int i; 
  
  f(af,mx,my,mz,hx,hy,hz,lx,mhx, mhy, mhz, mmhx,mmhy,mmhz,ffx,ffy,ffz);
  if(temp > 0.0 || af != 0)
    g(af,mx,my,mz,dwx,dwy,dwz,lx,Mhx, Mhy, Mhz, Mmhx,Mmhy,Mmhz,ggx,ggy,ggz);
  else
    for(i=0;i<lx;i++)
      ggx[i]=ggy[i]=ggz[i]=0;
  
  for(i=0;i<lx;i++ ){
    magtx[i] = mx[i] + ffx[i]*dt + ggx[i];
    magty[i] = my[i] + ffy[i]*dt + ggy[i];
    magtz[i] = mz[i] + ffz[i]*dt + ggz[i];
  }
  
  f(af,magtx,magty,magtz,hx,hy,hz,lx,mhx, mhy, mhz, mmhx,mmhy,mmhz,fftx,ffty,fftz);
  
  if(temp > 0.0 || af != 0)
    g(af,magtx,magty,magtz,dwx,dwy,dwz,lx,Mhx, Mhy, Mhz, Mmhx,Mmhy,Mmhz,ggtx,ggty,ggtz);
  else
    for(i=0;i<lx;i++)
      ggtx[i]=ggty[i]=ggtz[i]=0;
  
  
  for(i=0;i<lx;i++ ){
    magnewx[i] = mx[i] + 0.5*(ffx[i]+fftx[i])*dt + 0.5*(ggtx[i]+ggx[i]);
    magnewy[i] = my[i] + 0.5*(ffy[i]+ffty[i])*dt + 0.5*(ggty[i]+ggy[i]);
    magnewz[i] = mz[i] + 0.5*(ffz[i]+fftz[i])*dt + 0.5*(ggtz[i]+ggz[i]);
  }
  
  for(i=0;i<lx;i++ ){
    mx[i] = magnewx[i];
    my[i] = magnewy[i];
    mz[i] = magnewz[i];
  }
  return;
}   
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void anima(int animacao, double *rx, double *ry, double *rz, int plot,int lx,double *mx, double *my, double *mz)
{
  long int t;
  int i,j,k;
  
  if(animacao == 1){ 
    
    if(t%corte==0){
      printf("set xrange[%d:%d]\n",-1,lx+1);
      printf("set yrange[%d:%d]\n",-1,1);
      printf("set zrange[-1:%d]\n",1);
      printf("set cbrange [-1:1]\n");
      printf("unset grid\n");
      printf("unset key\n");
      printf("set palette defined ( 0 0 1 0, 0.3333 0 0 1, 0.6667 1 0 0,1 1 0.6471 0 )\n");
      
      if(plot == 1)
	printf("sp '-' u 1:2:3:4:5:6 with vectors head size 0.08,20,60 filled lc palette \n");
      if(plot == 2)
	printf("sp '-' u 1:2:3:4:5:6 with vectors head size 0.08,20,10 filled \n");
      if(plot == 0)
	printf("p '-' u 1:2:3:4:5  with vectors head size 0.5,15,60 filled lc palette\n");
      
      
      for(i=0;i<lx;i++ ){	    
	puts("ok");
	printf("%d %d %f %f %f\n",i,0,mx[i],my[i],mz[i]);
	//	    printf("%d %f %f %f\n",i,mx[i],my[i],mz[i]);
	
      }
      printf("e\n"); 
    }
    return;
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void orientacao(double *mx, double *my, double *mz){
  int i;
  
  if(direction == 1){ // DISTRIBUIÇÃO INICIAL DAS PARTÍCULAS
    for(i=0;i<Lx;i++){
      
      double ang2=FRANDOM*2*PI;	    	//para que a Magnetização inicie com
      double ang1=(2*FRANDOM-1)*PI;	        //orientação aleatória
      
      mx[i] = sin(ang1) * cos(ang2);
      my[i] = sin(ang1) * sin(ang2);
      mz[i] = cos(ang1);
    }
  }
  else{
    for(i=0;i<Lx;i++){
      
      mx[i] = 1.0;
      my[i] = 0.0;
      mz[i] = 0.0;
    }
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void ordem_eixos_anis_uni(int E_facil_u, double *Eax_x,double *Eax_y,double *Eax_z, int lx){
  int i;
  if (E_facil_u == 1){ //DIREÇÃO DOS EIXOS DE ANISOTROPIA UNIAXIAL DAS PARTÍCULAS
    
    for(i=0; i<lx; i++){
      double fi   = FRANDOM*2*PI;
      double teta = (2*FRANDOM-1)*PI;  
      
      Eax_x[i] = sin(teta) * cos(fi);
      Eax_y[i] = sin(teta) * sin(fi);
      Eax_z[i] = cos(teta);
      
      double mod = sqrt(pow(Eax_x[i],2) + pow(Eax_y[i],2) + pow(Eax_z[i],2));
      
      Eax_x[i] = Eax_x[i]/mod; 
      Eax_y[i] = Eax_y[i]/mod;
      Eax_z[i] = Eax_z[i]/mod;
    }
  }
  else{
    for (i=0; i<lx; i++){
      Eax_x[i] = 1.0;
      Eax_y[i] = 0.0;
      Eax_z[i] = 0.0;
      
      double mod = sqrt(pow(Eax_x[i],2) + pow(Eax_y[i],2) + pow(Eax_z[i],2));
      
      Eax_x[i] = Eax_x[i]/mod; 
      Eax_y[i] = Eax_y[i]/mod;
      Eax_z[i] = Eax_z[i]/mod;
    }
  }
}

/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void ordem_eixos_anis_cub(int E_facil_c, double *Eax_xc,double *Eax_yc,double *Eax_zc, int lx){
  int i;
  if (E_facil_c == 1){ //DIREÇÃO DOS EIXOS DE ANISOTROPIA CÚBICA DAS PARTÍCULAS
    
    for(i=0; i<lx; i++){
      double fi   = FRANDOM*2*PI;
      double teta = (2*FRANDOM-1)*PI;  
      
      Eax_xc[i] = sin(teta) * cos(fi);
      Eax_yc[i] = sin(teta) * sin(fi);
      Eax_zc[i] = cos(teta);
      
      double mod = sqrt(pow(Eax_xc[i],2) + pow(Eax_yc[i],2) + pow(Eax_zc[i],2));
      
      Eax_xc[i] = Eax_xc[i]/mod; 
      Eax_yc[i] = Eax_yc[i]/mod;
      Eax_zc[i] = Eax_zc[i]/mod;
    }
  }
  else{
    for (i=0; i<lx; i++){
      Eax_xc[i] = 1.0;
      Eax_yc[i] = 1.0;
      Eax_zc[i] = 1.0;
      
      double mod = sqrt(pow(Eax_xc[i],2) + pow(Eax_yc[i],2) + pow(Eax_zc[i],2));
      
      Eax_xc[i] = Eax_xc[i]/mod; 
      Eax_yc[i] = Eax_yc[i]/mod;
      Eax_zc[i] = Eax_zc[i]/mod;
    }
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void ordem_const_anis_uni(int Anis_u, int Anis_c, int Auni,int lx,double kku, double kkc,double *KKu, double *KKc){
  double ku_medio,kc_medio;
  int i;
  if (Anis_u == 1 || Anis_c == 1){
    if (Auni == 1){ // CONSTANTES DE ANISOTROPIA
      for(i=0; i<lx;i++){
	KKu[i] = kku*lognormal();
	KKc[i] = kkc*lognormal();
	ku_medio += KKu[i]/Lx;
	kc_medio += KKc[i]/Lx;
      }
    }
    else{
      for(i=0; i<lx; i++){
	KKu[i] = kku;
	KKc[i] = kkc;
	kc_medio += KKc[i]/Lx;
      }
    }
  }
  else{
    for(i=0; i<lx; i++){
      KKu[i] = 0;
      KKc[i] = 0;
      ku_medio = 1.0;
      kc_medio = 1.0;
    }
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void ordem_momentos(int Direction, int lx,double *mx,double *my,double *mz){
  int i;
  if(Direction == 1){ // DISTRIBUIÇÃO INICIAL DAS PARTÍCULAS
    for(i=0;i<lx;i++){
      
      double ang2=FRANDOM*2*PI;	    	//para que a Magnetização inicie com
      double ang1=(2*FRANDOM-1)*PI;	        //orientação aleatória
      
      mx[i] = sin(ang1) * cos(ang2);
      my[i] = sin(ang1) * sin(ang2);
      mz[i] = cos(ang1);
    }
  }
  else{
    for(i=0;i<lx;i++){
      mx[i] = 1.0;
      my[i] = 0.0;
      mz[i] = 0.0;
    }
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void ordem_posicao(int Dvar, int lx,double Dx, double *RR){
  int i;
  if(Dvar == 1){
    for (i=0;i<lx;i++){
      double dd = 1e-9*lognormal();
      if(i==0) dd = 0;
      RR[i] = Dx*i - Dx*dd;
    }
  }
  else{
    for (i=0;i<lx;i++)
      RR[i] = i*Dx;
  }
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void zerar_campos(int lx, double *hef_x,double *hef_y,double *hef_z, double *hz_x,double *hz_y,double *hz_z,double *hd_x,double *hd_y,double *hd_z, double *hau_x,double *hau_y,double *hau_z,double *hac_x,double *hac_y,double *hac_z){        
  int i;
  for (i=0;i<lx;i++){
    hef_x[i] = 0; hef_y[i] = 0; hef_z[i] = 0;
    hz_x[i]  = 0; hz_y[i]  = 0; hz_z[i]  = 0;
    hd_x[i]  = 0; hd_y[i]  = 0; hd_z[i]  = 0;
    hau_x[i] = 0; hau_y[i] = 0; hau_z[i] = 0; 
    hac_x[i] = 0; hac_y[i] = 0, hac_z[i] = 0;
  }
  return;	
}
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------------------------------------------------------------------------------------*/  
void campo_efetivo(int I, double *hef_x,double *hef_y,double *hef_z, double *hz_x,double *hz_y,double *hz_z,double *hd_x,double *hd_y,double *hd_z, double *hau_x,double *hau_y,double *hau_z,double *hac_x,double *hac_y,double *hac_z){
  //     int i;
  //  for(I=0;I<lx;I++){
  hef_x[I] = hau_x[I] + hac_x[I] + hz_x[I] +  hd_x[I];
  hef_y[I] = hau_y[I] + hac_y[I] + hz_y[I] +  hd_y[I];
  hef_z[I] = hau_z[I] + hac_z[I] + hz_z[I] +  hd_z[I];
  
  //    }
}
