         PROGRAM logistic

    

        real*8   z,zz,lnq,u1,u2,q,ran3,mu,sigma,kappa,a
        integer*4  d,i,j,N,seed
	character*66 file1




	
	write(*,*) 'number of deviates  N= ?'
	read(*,*) N

	write(*,*) 'alpha value= ?'
	read(*,*) a


	write(*,*) 'kappa value = ?'
	read(*,*) kappa

        d= 1
        q= 1+ (a*kappa)/(1+d*kappa)
        write(*,*) 'q value=', q



	write(*,*) 'mean (mu) = ?'
	read(*,*) mu

	write(*,*) 'sigma = ?'
	read(*,*) sigma



	write(*,*) 'seed= ?'
	read(*,*) seed


        write(*,*) 'filename writing= ?'
	read(*,*) file1
        open(unit=8,file=file1)
 

         pi=3.14159265358979324d0

c         q_prime=(q+1)/(3-q)
c
c         qvar=1.d0/(betaq*(3-q))
         

         do i=1, N
           u1=ran3(seed)
           u2=ran3(seed)
           u1= u1**(-2)
           lnq= ((u1**kappa-1)/kappa)
           z=   sqrt(lnq)*cos(2*pi*u2)
           zz=  qmu + sigma*z
           write(8,*) zz 
         enddo
 

c         write(*,*) 'q=',q, '  q_prime =',q_prime
   

         close(8)
	 

             stop 
             end
 

       FUNCTION ran3(idum)
      INTEGER*4 idum
      INTEGER*4 MBIG,MSEED,MZ
C     REAL*8 MBIG,MSEED,MZ
      REAL*8 ran3,FAC
      PARAMETER (MBIG=1000000000,MSEED=161803398,MZ=0,FAC=1.0d0/MBIG)
C     PARAMETER (MBIG=4000000.,MSEED=1618033.,MZ=0.,FAC=1./MBIG)
      INTEGER*4 i,iff,ii,inext,inextp,k
      INTEGER*4 mj,mk,ma(55)
C     REAL*8 mj,mk,ma(55)
      SAVE iff,inext,inextp,ma
      DATA iff /0/
      if(idum.lt.0.or.iff.eq.0)then
        iff=1
        mj=MSEED-iabs(idum)
        mj=mod(mj,MBIG)
        ma(55)=mj

        mk=1
        do 11 i=1,54
          ii=mod(21*i,55)
          ma(ii)=mk
          mk=mj-mk
          if(mk.lt.MZ)mk=mk+MBIG
          mj=ma(ii)
11      continue
        do 13 k=1,4
          do 12 i=1,55
            ma(i)=ma(i)-ma(1+mod(i+30,55))
            if(ma(i).lt.MZ)ma(i)=ma(i)+MBIG
12        continue
13      continue
        inext=0
        inextp=31
        idum=1
      endif
      inext=inext+1
      if(inext.eq.56)inext=1
      inextp=inextp+1
      if(inextp.eq.56)inextp=1
      mj=ma(inext)-ma(inextp)

      if(mj.lt.MZ)mj=mj+MBIG
      ma(inext)=mj
      ran3=mj*FAC
      return
      END
