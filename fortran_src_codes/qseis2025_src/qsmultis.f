      subroutine qsmultis(grnexist)
      implicit none
      logical*4 grnexist
      include 'qsglobal.h'
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     this program calculates convolution integral (summation)         c
c     of discrete sources to model the seismic ground motion of        c
c     an earthquake with arbitrary moment tensor distribution          c
c                                                                      c
c     Modified to support Stress, Strain and Rotation synthesis        c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      integer*4 ir,lf,icmp,istp,unit
      real*8 az1,az2
      real*8 weight(0:5)
      real*8 t(2*nfmax),y1(nrmax),y2(nrmax)
      character*110 textline
      
      ! Temporary complex weights for synthesis
      complex*16 w_psv_1, w_psv_2, w_psv_3, w_psv_4
      complex*16 w_sh_2, w_sh_3
c
      real*8 deg2rad
      data deg2rad/1.745329251994328d-02/
c
      if(grnexist)then
        do istp=1,4
          ! Loop extended to 19 to cover stress/strain/rotation
          do icmp=1,19
            ! Check for SH components (3,7,10,13,16,17,18) which are zero for 
            ! isotropic sources (EX=1, CL=4, i.e., ms(istp)=0)
            if(ms(istp).eq.0 .and. (icmp.eq.3 .or. icmp.eq.7 .or. 
     &         icmp.eq.10 .or. icmp.eq.13 .or. icmp.eq.16 .or. 
     &         icmp.eq.17 .or. icmp.eq.18)) then
              do lf=1,nf
                do ir=1,nr
                  grns(lf,icmp,ir,istp)=(0.d0,0.d0)
                enddo
              enddo
            else
              unit=istp*20+icmp
              open(unit,file=outfile(icmp,istp),status='old',err=90)
              read(unit,'(a1)')textline
              do lf=1,nf
                read(unit,*,end=100)t(2*lf-1),(y1(ir),ir=1,nr)
                read(unit,*,end=100)t(2*lf),(y2(ir),ir=1,nr)
                do ir=1,nr
                  grns(lf,icmp,ir,istp)=dcmplx(y1(ir),y2(ir))
                enddo
              enddo
100           close(unit)
              goto 95
              
90            continue
              ! If file does not exist, zero out the component
              do lf=1,nf
                do ir=1,nr
                  grns(lf,icmp,ir,istp)=(0.d0,0.d0)
                enddo
              enddo
95            continue
            endif
          enddo
        enddo
      endif
c
      weight(0)=(mtensor(1)+mtensor(2)+mtensor(3))/3.d0
      weight(1)=mtensor(4)
      weight(2)=mtensor(6)
      weight(3)=mtensor(3)-weight(0)
      weight(4)=0.5d0*(mtensor(1)-mtensor(2))
      weight(5)=mtensor(5)
c
      do ir=1,nr
        az1=azimuth(ir)*deg2rad
        az2=2.d0*az1
        
        ! Pre-calculate weights for P-SV and SH
        ! P-SV weights
        w_psv_1 = dcmplx(weight(0), 0.d0)
        w_psv_2 = dcmplx(weight(1)*dsin(az2)+weight(4)*dcos(az2), 0.d0)
        w_psv_3 = dcmplx(weight(2)*dcos(az1)+weight(5)*dsin(az1), 0.d0)
        w_psv_4 = dcmplx(weight(3), 0.d0)
        
        ! SH weights
        w_sh_2  = dcmplx(weight(1)*dcos(az2)-weight(4)*dsin(az2), 0.d0)
        w_sh_3  = dcmplx(weight(2)*dsin(az1)-weight(5)*dcos(az1), 0.d0)

        do lf=1,nf
          do icmp=1,19
            ! === P-SV Type Components ===
            ! 1:uz, 2:ur, 4:vol
            ! 5:ezz, 6:ezr, 8:err, 9:ett
            ! 11:szz, 12:szr, 14:srr, 15:stt
            ! 19:ot (rotation theta)
            if (icmp.eq.1.or.icmp.eq.2.or.icmp.eq.4.or. 
     &          icmp.eq.5.or.icmp.eq.6 .or.icmp.eq.8.or.icmp.eq.9.or.
     &          icmp.eq.11.or.icmp.eq.12.or.icmp.eq.14.or.icmp.eq.15.or.
     &          icmp.eq.19) then
              grns(lf,icmp,ir,7) = grns(lf,icmp,ir,1) * w_psv_1
     &                           + grns(lf,icmp,ir,2) * w_psv_2
     &                           + grns(lf,icmp,ir,3) * w_psv_3
     &                           + grns(lf,icmp,ir,4) * w_psv_4
            
            ! === SH Type Components ===
            ! 3:ut
            ! 7:ezt, 10:ert
            ! 13:szt, 16:srt
            ! 17:oz, 18:or
            else if (icmp.eq.3 .or. icmp.eq.7 .or. icmp.eq.10 .or.
     &               icmp.eq.13 .or. icmp.eq.16 .or. 
     &               icmp.eq.17 .or. icmp.eq.18) then
              grns(lf,icmp,ir,7) = grns(lf,icmp,ir,2) * w_sh_2
     &                           + grns(lf,icmp,ir,3) * w_sh_3
            endif
          enddo
        enddo
      enddo
      
      if(grnexist)then
        do icmp=1,19
          if(fsel(icmp,7).eq.1)then
            unit=40+icmp
            open(unit,file=outfile(icmp,7),status='unknown')
            write(unit,'(a,$)')'   T_sec    '
            do ir=1,nr-1
              write(unit,'(a4,a1,a1,2a4,$)')'   ',
     &             varbtxt,comptxt(icmp),rcvtxt(ir),'   '
            enddo
            write(unit,'(a4,a1,a1,2a4)')'   ',
     &             varbtxt,comptxt(icmp),rcvtxt(nr),'   '
            do lf=1,nf
              write(unit,'(f12.5,$)')t(2*lf-1)
              do ir=1,nr-1
                write(unit,'(E12.4,$)')dreal(grns(lf,icmp,ir,7))
              enddo
              write(unit,'(E12.4)')dreal(grns(lf,icmp,nr,7))
c
              write(unit,'(f12.5,$)')t(2*lf)
              do ir=1,nr-1
                write(unit,'(E12.4,$)')dimag(grns(lf,icmp,ir,7))
              enddo
              write(unit,'(E12.4)')dimag(grns(lf,icmp,nr,7))
            enddo
            close(unit)
          endif
        enddo
      endif
c
      return
      end
