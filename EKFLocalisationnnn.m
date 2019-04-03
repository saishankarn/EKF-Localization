
function EKFLocalisation
rhol=0;
close all;
clear all;
q=1;
w=0;
pos=0;
zer=0;
neg=0;
negerrk=0;
error=zeros(100,1);
global xVehicleTrue;global Map;global RTrue;global UTrue;global nSteps;
for q=1:100
    
errpos=0;
errneg=0;
covnegerr=0
nSteps = 100;
nFeatures = 1;
MapSize = 20;
alpha=0;
Map = [0;-0.75];
UTrue = diag([0.01,0.01,1*pi/180]).^2;
RTrue = [100.0,0;0,36*pi*pi/(180*180)];
sigmaR=10;
sigmaTheta=6*pi/180;
UEst = 1.0*UTrue;
REst = 1.0*RTrue;

xVehicleTrue = [0;0;-pi/2];

%initial conditions - no map:

xEstcorr =[];
PEstcorr = [];
xEst =[];
PEst = [];

MappedFeatures = NaN*zeros(nFeatures,2);

%storage:
PStore = NaN*zeros(nFeatures,nSteps);
XErrStore = NaN*zeros(nFeatures,nSteps);


%initial graphics - plot true map
 hold on; grid off; axis equal;
plot(Map(1,:),Map(2,:),'g*');hold on;
%set(gcf,'doublebuffer','on');
%hObsLine = line([0,0],[0,0]);
%set(hObsLine,'linestyle',':');

i=0;
b=0;

u=0;
for k = 2:nSteps
    
    %do world iteration
    SimulateWorld(k);
    
    %simple prediction model:
    xPredcorr = xEstcorr;
    PPredcorr = PEstcorr;
    
         xPred = xEst;
    PPred = PEst;
          
        
    %observe a randomn feature
    [z,iFeature] = GetObservation(k);
    
    if(~isempty(z))
        
        %have we seen this feature before?
        if( ~isnan(MappedFeatures(iFeature,1)))
            
            %predict observation: find out where it is in state vector
            FeatureIndex = MappedFeatures(iFeature,1);
            xFeaturecorr = xPredcorr(FeatureIndex:FeatureIndex+1);
            zPredcorr = DoObservationModel(xVehicleTrue,xFeaturecorr);
            
              xFeature = xPred(FeatureIndex:FeatureIndex+1);
            zPred = DoObservationModel(xVehicleTrue,xFeature);
            
            % get observation Jacobians
            [jHxvcorr,jHxfcorr] = GetObsJacs(xVehicleTrue,xFeaturecorr);
            
          

            
            % fill in state jacobian
            jHcorr = zeros(2,length(xEstcorr));
            jHcorr(:,FeatureIndex:FeatureIndex+1) = jHxfcorr;
            
            
            
           
            [jHxv,jHxf] = GetObsJacs(xVehicleTrue,xFeature);
            
            % fill in state jacobian
            jH = zeros(2,length(xEst));
            jH(:,FeatureIndex:FeatureIndex+1) = jHxf;
            
            
            zmean = DoObservationModel(xVehicleTrue,Map(:,iFeature));
            zmean(2)=AngleWrap(zmean(2));
            
            %do Kalman update:
            
                pr=zmean(1);
            if(-pi<zmean(2)<-168.75*pi/180)
                ptangle=zmean(2)+pi;
                ptangle=abs(ptangle);
            elseif(-168.75*pi/180<zmean(2)<-146.25*pi/180)
                ptangle=zmean(2)+157.5*pi/180;
                ptangle=abs(ptangle);
            elseif(-146.25*pi/180<zmean(2)<-123.75*pi/180)
                ptangle=zmean(2)+135*pi/180;
                ptangle=abs(ptangle);
            elseif(-123.75*pi/180<zmean(2)<-101.25*pi/180)
                ptangle=zmean(2)+112.5*pi/180;
                ptangle=abs(ptangle);
            elseif(-101.25*pi/180<zmean(2)<-78.75*pi/180)
                ptangle=zmean(2)+90*pi/180;
                ptangle=abs(ptangle);
            elseif(-78.75*pi/180<zmean(2)<-56.25*pi/180)
                ptangle=zmean(2)+67.5*pi/180;
                ptangle=abs(ptangle);
            elseif(-56.25*pi/180<zmean(2)<-33.75*pi/180)
                ptangle=zmean(2)+45*pi/180;
                ptangle=abs(ptangle); 
            elseif(-33.75*pi/180<zmean(2)<-11.25*pi/180)
                ptangle=zmean(2)+22.5*pi/180;
                ptangle=abs(ptangle);  
            elseif(-11.25*pi/180<zmean(2)<+11.25*pi/180)
                ptangle=zmean(2)+0*pi/180;
                ptangle=abs(ptangle);
            elseif(+11.25*pi/180<zmean(2)<33.75*pi/180)
                ptangle=zmean(2)-22.5*pi/180;
                ptangle=abs(ptangle);
            elseif(33.75*pi/180<zmean(2)<56.25*pi/180)
                ptangle=zmean(2)-45*pi/180;
                ptangle=abs(ptangle); 
            elseif(56.25*pi/180<zmean(2)<78.75*pi/180)
                ptangle=zmean(2)-67.5*pi/180;
                ptangle=abs(ptangle);
            elseif(78.75*pi/180<zmean(2)<101.25*pi/180)
                ptangle=zmean(2)-90*pi/180;
                ptangle=abs(ptangle);
            elseif(101.25*pi/180<zmean(2)<123.75*pi/180)
                ptangle=zmean(2)-112.5*pi/180;
                ptangle=abs(ptangle);
            elseif(123.75*pi/180<zmean(2)<146.25*pi/180)
                ptangle=zmean(2)-135*pi/180;
                ptangle=abs(ptangle);
            elseif(146.25*pi/180<zmean(2)<168.75*pi/180)
                ptangle=zmean(2)-157.5*pi/180;
                ptangle=abs(ptangle);
            elseif(168.75*pi/180<zmean(2)<pi)
                ptangle=zmean(2)-pi;
                ptangle=abs(ptangle);
            end     
            
          
 
            PMean=pressuree(pr,ptangle);
            
            
            
            
            
               percorr=zPredcorr(1);
            if(-pi<zPredcorr(2)<-168.75*pi/180)
                petanglecorr=zPredcorr(2)+pi;
                petanglecorr=abs(petanglecorr);
            elseif(-168.75*pi/180<zPredcorr(2)<-146.25*pi/180)
                petanglecorr=zPredcorr(2)+157.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(-146.25*pi/180<zPredcorr(2)<-123.75*pi/180)
                petanglecorr=zPredcorr(2)+135*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(-123.75*pi/180<zPredcorr(2)<-101.25*pi/180)
                petanglecorr=zPredcorr(2)+112.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(-101.25*pi/180<zPredcorr(2)<-78.75*pi/180)
                petanglecorr=zPredcorr(2)+90*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(-78.75*pi/180<zPredcorr(2)<-56.25*pi/180)
                petanglecorr=zPredcorr(2)+67.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(-56.25*pi/180<zPredcorr(2)<-33.75*pi/180)
                petanglecorr=zPredcorr(2)+45*pi/180;
                petanglecorr=abs(petanglecorr); 
            elseif(-33.75*pi/180<zPredcorr(2)<-11.25*pi/180)
                petanglecorr=zPredcorr(2)+22.5*pi/180;
                petanglecorr=abs(petanglecorr);  
            elseif(-11.25*pi/180<zPredcorr(2)<+11.25*pi/180)
                petanglecorr=zPredcorr(2)+0*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(+11.25*pi/180<zPredcorr(2)<33.75*pi/180)
                petanglecorr=zPredcorr(2)-22.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(33.75*pi/180<zPredcorr(2)<56.25*pi/180)
                petanglecorr=zPredcorr(2)-45*pi/180;
                petanglecorr=abs(petanglecorr); 
            elseif(56.25*pi/180<zPredcorr(2)<78.75*pi/180)
                petanglecorr=zPredcorr(2)-67.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(78.75*pi/180<zPredcorr(2)<101.25*pi/180)
                petanglecorr=zPredcorr(2)-90*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(101.25*pi/180<zPredcorr(2)<123.75*pi/180)
                petanglecorr=zPredcorr(2)-112.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(123.75*pi/180<zPredcorr(2)<146.25*pi/180)
                petanglecorr=zPredcorr(2)-135*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(146.25*pi/180<zPredcorr(2)<168.75*pi/180)
                petanglecorr=zPredcorr(2)-157.5*pi/180;
                petanglecorr=abs(petanglecorr);
            elseif(168.75*pi/180<zPredcorr(2)<pi)
                petanglecorr=zPredcorr(2)-pi;
                petanglecorr=abs(petanglecorr);
            end     
            
            
            
           PEsti=pressuree(percorr,petanglecorr);
          
          
          
           
             pmr=z(1);
            if(-pi<z(2)<-168.75*pi/180)
                pmtangle=z(2)+pi;
                pmtangle=abs(pmtangle);
            elseif(-168.75*pi/180<z(2)<-146.25*pi/180)
                pmtangle=z(2)+157.5*pi/180;
                pmtangle=abs(pmtangle);
            elseif(-146.25*pi/180<z(2)<-123.75*pi/180)
                pmtangle=z(2)+135*pi/180;
                pmtangle=abs(pmtangle);
            elseif(-123.75*pi/180<z(2)<-101.25*pi/180)
                pmtangle=z(2)+112.5*pi/180;
                pmtangle=abs(pmtangle);
            elseif(-101.25*pi/180<z(2)<-78.75*pi/180)
                pmtangle=z(2)+90*pi/180;
                pmtangle=abs(pmtangle);
            elseif(-78.75*pi/180<z(2)<-56.25*pi/180)
                pmtangle=z(2)+67.5*pi/180;
                pmtangle=abs(pmtangle);
            elseif(-56.25*pi/180<z(2)<-33.75*pi/180)
                pmtangle=z(2)+45*pi/180;
                pmtangle=abs(pmtangle); 
            elseif(-33.75*pi/180<z(2)<-11.25*pi/180)
                pmtangle=z(2)+22.5*pi/180;
                pmtangle=abs(pmtangle);  
            elseif(-11.25*pi/180<z(2)<+11.25*pi/180)
                pmtangle=z(2)+0*pi/180;
                pmtangle=abs(pmtangle);
            elseif(+11.25*pi/180<z(2)<33.75*pi/180)
                pmtangle=z(2)-22.5*pi/180;
                pmtangle=abs(pmtangle);
            elseif(33.75*pi/180<z(2)<56.25*pi/180)
                pmtangle=z(2)-45*pi/180;
                pmtangle=abs(pmtangle); 
            elseif(56.25*pi/180<z(2)<78.75*pi/180)
                pmtangle=z(2)-67.5*pi/180;
                pmtangle=abs(pmtangle);
            elseif(78.75*pi/180<z(2)<101.25*pi/180)
                pmtangle=z(2)-90*pi/180;
                pmtangle=abs(pmtangle);
            elseif(101.25*pi/180<z(2)<123.75*pi/180)
                pmtangle=z(2)-112.5*pi/180;
                pmtangle=abs(pmtangle);
            elseif(123.75*pi/180<z(2)<146.25*pi/180)
                pmtangle=z(2)-135*pi/180;
                pmtangle=abs(pmtangle);
            elseif(146.25*pi/180<z(2)<168.75*pi/180)
                pmtangle=z(2)-157.5*pi/180;
     
                pmtangle=abs(pmtangle);
            elseif(168.75*pi/180<z(2)<pi)
                pmtangle=z(2)-pi;
                pmtangle=abs(pmtangle);
            end     
           
           
           PMeas=pressuree(pmr,pmtangle);
           
          
    
           
           edel=PMean-PEsti;
           mdel=PMean-PMeas;
           absedel=abs(edel);
           absmdel=abs(mdel);
           ethreshold=0.001;
           mthreshold=0.001;
           rho=0;
           if(absmdel>mthreshold)
           
           
           if(absedel<ethreshold)
         
             
             
                 zmean=zPredcorr;
                 b=b+1
            k;
                 rho=corrcalc(percorr,petanglecorr);
                
            
         
        
                  
             REstcorr=[sigmaR*sigmaR,rho*sigmaR*sigmaTheta;rho*sigmaR*sigmaTheta,sigmaTheta*sigmaTheta];
             e=eig(REstcorr);
            REstcorr=[e(2),0;0,e(1)];
             %psi=(2*rho*sigmaR*sigmaTheta/(sigmaR*sigmaR-sigmaTheta*sigmaTheta));
             %alpha1=0.5*atan(psi)
             [vec,val]=eig(REstcorr);
             % Get the index of the largest eigenvector
             [max_evc_ind_c, r] = find(val == max(max(val)));
             max_evc = vec(:, max_evc_ind_c);
             max_evl = max(max(val));
             % Get the smallest eigenvector and eigenvalue
             if(max_evc_ind_c == 1)
                 min_evl = max(val(:,2));
                 min_evc = vec(:,2);
             else
             min_evl = max(val(:,1));
             min_evc = vec(1,:);
             end
             alpha = atan2(max_evc(2), max_evc(1));
             rhol=rho;
             if(alpha>pi)
                alpha=pi-alpha;
            end
            y1=sqrt(e(2)/(sigmaR*sigmaR));
             y2=sqrt(e(1)/(sigmaTheta*sigmaTheta));
             l=z;
            Rot=[cos(alpha),-sin(alpha);sin(alpha),cos(alpha)];
            Scale=[y1,0;0,y2];
            z1=z-zmean;
            z1=Rot*z1;
            z1=Scale*z1;
            T=Rot;
            jHcorr=T*jHcorr;
            Innov = z1;
            Innov(2) = AngleWrap(Innov(2));
            S = jHcorr*PPredcorr*jHcorr'+REstcorr;
            W = PPredcorr*jHcorr'*inv(S);
            xEstcorr = xPredcorr+ W*Innov;
            PEstcorr = PPredcorr-W*S*W';
            PEstcorr = 0.5*(PEstcorr+PEstcorr');
           
            
            
             end
           end
               
            
            if(rho==0)
            
            REstcorr=[sigmaR*sigmaR,rho*sigmaR*sigmaTheta;rho*sigmaR*sigmaTheta,sigmaTheta*sigmaTheta];
            Innov = z-zPredcorr;
            Innov(2) = AngleWrap(Innov(2));
            S = jHcorr*PPredcorr*jHcorr'+REstcorr;
            W = PPredcorr*jHcorr'*inv(S);
            xEstcorr = xPredcorr+ W*Innov;
            PEstcorr = PPredcorr-W*S*W';
            PEstcorr = 0.5*(PEstcorr+PEstcorr');
            end
            Innov = z-zPred;
            Innov(2) = AngleWrap(Innov(2));
            
            S = jH*PPred*jH'+REst;
            W = PPred*jH'*inv(S);
            xEst = xPred+ W*Innov;
            
            PEst = PPred-W*S*W';
            
   else
                  
                  nStates = length(xEstcorr); 
                  xFeaturecorr = xVehicleTrue(1:2)+ [z(1)*cos(z(2)+xVehicleTrue(3));z(1)*sin(z(2)+xVehicleTrue(3))];
                  xEstcorr = [xEstcorr;xFeaturecorr];
                  [jGxvcorr, jGzcorr] = GetNewFeatureJacs(xVehicleTrue,z);
                  M = [eye(nStates), zeros(nStates,2);
                  zeros(2,nStates)  , jGzcorr];
                  PEstcorr = M*blkdiag(PEstcorr,REst)*M';
                  MappedFeatures(iFeature,:) = [length(xEstcorr)-1, length(xEstcorr)];

                   nStates = length(xEst); 
            
            xFeature = xVehicleTrue(1:2)+ [z(1)*cos(z(2)+xVehicleTrue(3));z(1)*sin(z(2)+xVehicleTrue(3))];
            xEst = [xEst;xFeature];
            [jGxv, jGz] = GetNewFeatureJacs(xVehicleTrue,z);
            
            M = [eye(nStates), zeros(nStates,2);% note we don't use jacobian w.r.t vehicle
                zeros(2,nStates)  , jGz];
            
            PEst = M*blkdiag(PEst,REst)*M';
            
            %remember this feature as being mapped we store its ID and position in the state vector
            MappedFeatures(iFeature,:) = [length(xEst)-1, length(xEst)];
                  
            end
    else             
        end  
          
           
          

               
                 
                  colorstring = 'br';
  
                 
                  plot(xEstcorr(1),k, 'Color', colorstring(2));
                  hold on
                 
               
                 
                  xEstcorr
                  xEst
                 
                     ecorr= sqrt(((xEstcorr(1)+0)^2)+((xEstcorr(2)+0.75)^2));
                     e=sqrt(((xEst(1)+0)^2)+((xEst(2)+0.75)^2));
                     err=ecorr-e;
                    e1=eig(PEstcorr);
                    e2=eig(PEst);
                    errcorr=e1(1)*e1(2);
                    eruncorr=e2(1)*e2(2);
                    areaerr=errcorr-eruncorr;
                    if(areaerr<0)
                        negerrk=negerrk+1
                    end
                    error(k,1)=err
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
%if(abs(err)>0.04)
if(err<0)
    neg=neg+1;
end
if(err>0)
    pos=pos+1;
end

%end


end
neg
pos
zer  
error;
negerrk;
covnegerr;
%errneg
%errpos
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function P = pressuree(r,theta)
ka = 17.6;
fa = 0.006516;
beta = 970;
P = beta*(4*0.006516/r^2)*(besselj(1,ka*sin(theta))/(ka*sin(theta)))^2;
%P=P+0.0001*randn(1);
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rho=corrcalc(rmean,tmean)
fa=0.006516;
beta=970;
ka=17.6;
%%
% 
% $$e^{\pi i} + 1 = 0$$
% 
t=ka*tmean;
m=ka/2;
l=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Pcorr=pressuree(rmean,tmean);
cons=sqrt(beta*fa/Pcorr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


rt=tmean+(12*pi/180).*randn(100,1);
x=rt*m;
ftheta=1-(x.^2)*(1/2)+(x.^4)*(1/12)-(x.^6)*(1/144)+(x.^8)*(1/2880)-(x.^10)*(1/86400);

rr=cons*ftheta;
A=[rr rt];
R = corrcoef(A);

rho=R(1,2);



end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [z,iFeature] = GetObservation(k)
global Map;global xVehicleTrue;global RTrue;global nSteps;
%choose a random feature to see from True Map
iFeature = ceil(size(Map,2)*rand(1));
z = DoObservationModel(xVehicleTrue,Map(:,iFeature))+sqrt(RTrue)*randn(2,1);
z(2) = AngleWrap(z(2));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z] = DoObservationModel(xVeh, xFeature)
Delta = xFeature-xVeh(1:2);
z = [norm(Delta);
    atan2(Delta(2),Delta(1))-xVeh(3)];
z(2) = AngleWrap(z(2));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SimulateWorld(k)
global xVehicleTrue;
u = GetRobotControl(k);
xVehicleTrue = tcomp(xVehicleTrue,u);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [jHxv,jHxf] = GetObsJacs(xPred, xFeature)
jHxv = zeros(2,3);jHxf = zeros(2,2);
Delta = (xFeature-xPred(1:2));
r = norm(Delta);
jHxv(1,1) = -Delta(1) / r;
jHxv(1,2) = -Delta(2) / r;
jHxv(2,1) = Delta(2) / (r^2);
jHxv(2,2) = -Delta(1) / (r^2);
jHxv(2,3) = -1;
jHxf(1:2,1:2) = -jHxv(1:2,1:2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [jGx,jGz] = GetNewFeatureJacs(Xv, z);
x = Xv(1,1);
y = Xv(2,1);
theta = Xv(3,1);
r = z(1);
bearing = z(2);
jGx = [ 1   0   -r*sin(theta + bearing);
        0   1   r*cos(theta + bearing)];
jGz = [ cos(theta + bearing) -r*sin(theta + bearing);
        sin(theta + bearing) r*cos(theta + bearing)];
end     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function u = GetRobotControl(k)
global nSteps;
%u = [0; 0.25 ; 0.3*pi/180*sin(3*pi*k/nSteps)];
u = [0; 0.15 ; 0.3*pi/180];
end