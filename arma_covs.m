function [gamma_arma] = arma_covs(lag,params,arma_part)
% This code is developed by Gy�rgy Inzelt. The license of this code is as follows
% -------------------------------------------------------------------------
% Copyright (c) 2011, Gy�rgy Inzelt
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
% --------------------------------------------------------------------------

     % Ian McLeod's method for calculating the analytical ACF for linear stationary ARMA(p,q) 
     % processes. 
     %
     % Reference: McLeod, Ian: Derivation of the Theoretical Autocovariance Function
     %                         of Autoregressive-Moving Average Time Series. 
     %                         Applied Statistics(1975) 24 No.2 p255-257 
     %
     % (C) Gy�rgy Inzelt 2011
     
     %% inputs
     % lag: an postive interger
     % params = [phi_1 ... phi_p ... theta_1 ... theta_q, sigma^2];
     % arma_part = [p, q];
     %% output
     % gamma_arma: k*1 vector where k=max(lag, p+1), the elements in
     % gamma_arma is [cov(0); cov(1); ...; cov(k-1)]
     %%
     wn_var = params(length(params));
     
     if  arma_part(1) == 0 && arma_part(2) == 0, phi = 0   ; theta =0;
       elseif arma_part(1) ==0 && arma_part(2) ~= 0 ,theta = params(1,1:1+arma_part(2)-1)   ; phi =0;
       elseif arma_part(1) ~=0 && arma_part(2) ~=0, phi = params(1,1:1+arma_part(1)-1);
              theta = params(1,1+arma_part(1):1+sum(arma_part)-1);
       elseif arma_part(1) ~=0 && arma_part(2)==0, phi = params(1,1:1+arma_part(1)-1); theta = 0;      
     end     
     
    gamma_arma = zeros(lag,1);    
   %ARMA(0,q) case 
    if arma_part(1) == 0 && arma_part(2)~=0
       for ii = 0:1:lag
          if ii == 0
             gamma_arma(ii+1) = (sum([1 theta].^2))*wn_var ;
          elseif ii > 0 && ii < arma_part(2)
             gamma_arma(ii+1) = (theta(ii) + sum( theta(ii+1:length(theta)).*theta(1:length(theta)-ii) ))*wn_var;
          elseif ii == arma_part(2)
             gamma_arma(ii+1) = theta(length(theta))*wn_var;
          elseif ii > arma_part(2)
             gamma_arma(ii+1) = 0; 
          end
       end
    %ARMA(0,0)-just for completeness   
    elseif arma_part(1)==0 && arma_part(2)==0  
     gamma_arma(1) = wn_var;
     gamma_arma(2:lags) = 0;
    %ARMA(p,0)   
    elseif arma_part(1) ~= 0 && arma_part(2)==0
       if arma_part(1)==1
          gamma_arma = (phi(1).^((0:1:lag)'))*(1/(1-phi(1)^2))*wn_var;%% 
       elseif arma_part(1) > 1    
          F = zeros(arma_part(1),arma_part(1));
          F(1,:) = phi;
          F(2:length(F),1:length(F)-1 )=eye(length(F)-1)  ;
          G = wn_var*inv(( eye((arma_part(1))^2) - kron(F,F)   )) ;
          gamma_arma(1:arma_part(1)) = G(1:arma_part(1),1);
          for jj = arma_part(1)+1:1:length(gamma_arma)
              gamma_arma(jj) =  phi*gamma_arma(jj-1:-1:jj-arma_part(1));
          end 
       end 
    %ARMA(p,q)  
    elseif arma_part(2) ~=0 && arma_part(2) ~=0    
      %calculating the cross-covariances
     [gamma_pacf, rhs, lhs]= arma_crosscov(phi,theta,arma_part,wn_var);
      gamma_arma(1:arma_part(1)+1,1)= lhs\rhs(1:arma_part(1)+1);
        if arma_part(1) >= arma_part(2)
          for jj = arma_part(1)+2:1:length(gamma_arma)
                  gamma_arma(jj) =  phi*gamma_arma(jj-1:-1:jj-arma_part(1));
          end 
        elseif arma_part(1) < arma_part(2) 
            for jj = arma_part(1)+2:1:length(gamma_arma)
               if jj  <=length(rhs)
                  gamma_arma(jj) =  phi*gamma_arma(jj-1:-1:jj-arma_part(1)) + rhs(jj);
               elseif jj  > length(rhs)   
                  gamma_arma(jj) =  phi*gamma_arma(jj-1:-1:jj-arma_part(1));
               end
            end
        end
    end  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[gamma_pacf, rhs, lhs]= arma_crosscov(phi,theta,arma_part,wn_var)
     gamma_pacf = zeros(arma_part(2)+1,1);
     rhs = zeros(max(arma_part(1)+1,arma_part(2)+1),1);
     lhs = zeros(arma_part(1)+1,arma_part(1)+1);
     
     gamma_pacf(1) = wn_var;
     %this could've been scripted in a more elegant manner
     for kk = 2:1:arma_part(2)+1
         for jj = 1:1:min(kk-1,arma_part(1))
         gamma_pacf(kk) = gamma_pacf(kk) + gamma_pacf(kk-jj)*phi(jj);    
         end    
         gamma_pacf(kk) = gamma_pacf(kk) +  theta(kk-1)*wn_var;    
     end     
     %rhs
     for ll = 1:1:max(arma_part(1)+1,arma_part(2)+1)
        if ll==1
           rhs(ll) = gamma_pacf(1) + theta*gamma_pacf(2:arma_part(2) + 1)  ;
        elseif ll > 1 && ll <= arma_part(2)+1 
           rhs(ll) = theta(ll-1:length(theta))*gamma_pacf(1:arma_part(2) + 2-ll) ;
        elseif ll > 1 && ll > arma_part(2)+1   
           rhs(ll) =0;
        end    
     end 
     %lhs
     lhs(1,:) =  [1 -phi];
     lhs(:,1) =  [1 -phi]';
     
     for ii = 2:1:arma_part(1)+1
         for jj = 2:1:arma_part(1)+1
            if ii-jj==0 && ii+jj-2 <= arma_part(1)
                phi_ij = 1;
                phi_ij2 = -phi(ii + jj -2);
            elseif ii-jj==0 && ii+jj-2 > arma_part(1)    
                phi_ij = 1;
                phi_ij2 = 0;
            elseif ii-jj < 0 && ii+jj-2 <=arma_part(1)
                phi_ij = 0;
                phi_ij2 = -phi(ii + jj -2);
            elseif ii-jj < 0 && ii+jj-2 > arma_part(1)    
                phi_ij = 0;
                phi_ij2 =0;
            elseif ii-jj > 0 && ii+jj-2 <= arma_part(1)    
                phi_ij = -phi(ii-jj);
                phi_ij2 = -phi(ii+jj-2);
            elseif ii-jj > 0 && ii+jj-2 > arma_part(1)       
                phi_ij = -phi(ii-jj);
                phi_ij2 = 0;
            end    
                lhs(ii,jj) = phi_ij + phi_ij2 ;         
         end
     end    
end