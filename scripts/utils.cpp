#include <stdio.h>
#include <cmath>



extern "C" void dist_circle (int nx, int ny, double xc, double yc, double * map) {
  int i,j;
  double x2,y2,dx,dy;
  for (i = 0; i < nx ; i++) {
    dx=double(i)-xc;
    x2=dx*dx;
      
    for (j = 0; j < ny ; j++) {
       dy=double(j)-yc;
       y2=dy*dy;
       map[i*nx+j] = sqrt( x2+y2 );
    };
  };
}




extern "C" void make_planet (int nx, int ny, double xc, double yc, double Rp, double * map) {
  int i,j;
  double x2,y2,dx,dy;
  for (i = 0; i < nx ; i++) {
    dx=double(i)-xc;
    x2=dx*dx;
      
    for (j = 0; j < ny ; j++) {
       dy=double(j)-yc;
       y2=dy*dy;
       if (sqrt( x2+y2) <=Rp) {
          map[i*nx+j] = 0.;
       } else {
	 map[i*nx+j]=1.;
       };
       
    };
  };
}




extern "C" void make_star (int nx, int ny, double xc, double yc, double Rstar,double u1, double u2, double * map)
{
  int i, j;
  double mu2;
  double Rs2=Rstar*Rstar;
  double *musq;
  long n=long(nx)*long(ny);
  musq=new double[n];

  for (i = 0; i < nx ; i++)
  {
     for (j = 0; j < ny ; j++)
     {
       musq[i*nx+j]=1.-( (double(i)-xc)*(double(i)-xc)+(double(j)-yc)*(double(j)-xc) )/Rs2;
     }
  }
  for (i = 0; i < n ; i++)
  {
     mu2=(musq[i]);
     if (mu2 >= 0.)
     {
       map[i]=1.-u1*(1.-sqrt(mu2)) - u2*(1.-mu2);
     }
     else
     {
        map[i]=0.;
     };  //else
     
  };//for i
  delete [] musq;
}


