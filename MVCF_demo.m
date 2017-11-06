%   The demo was written by Kun Zhan et al.
%   $Revision: 1.0.0.0 $  $Date: 2016/4/13 $ 10:09:46 $

%   Reference:
%   K. Zhan, J. Shi, J. Wang, H. Wang, and Y. Xie
%   "Adaptive structure concept factorization for multi-view clustering," 
%   Neural Computation, 2017.


addpath('./tools');
clear
dataset = load('M_3Sources.mat'); xp = -4.4;
% dataset = load('M_Cornell.mat'); xp = -2.6;
% dataset = load('M_Texas.mat'); xp = -2.6;
% dataset = load('M_Washington.mat'); xp = -2.8;
% dataset = load('M_Winsconsin.mat'); xp = -2.8;
% dataset = load('AwA.mat'); xp = -2.8;
% dataset = load('C101.mat'); xp = -4.2;
% dataset = load('Number_v1245.mat'); xp = -4.6;
% dataset = load('scene3v.mat'); xp = -4.8;
tic
X = dataset.X_train;
GT = dataset.truth;
lambda = 10; gamma = 10^xp;

%% initialization
point_num = size(X{1}, 2); % point number
view_num = length(X); % view number
nClass = length(unique(GT)); % cluster number
alpha = ones(view_num,1)/view_num; % initialize all alpha(i) as 1/view_num
% LCCFoptions
MVCFoptions = [];
MVCFoptions.WeightMode = 'Cosine';
MVCFoptions.bNormalized = 1;  
MVCFoptions.maxIter = 200;
MVCFoptions.weight = 'NCW';
MVCFoptions.alpha = 2; % 200 originally
MVCFoptions.KernelType = 'Linear'; % type of kernel: 'Linear'

U = cell(1,view_num); % U corresponds W in Cai's paper
V = cell(1,view_num); % V corresponds V in Cai's paper

P = cell(1,view_num);
sumP = zeros(point_num);

for i = 1:view_num
    X{i} = X{i}/sum(sum(X{i})); % normalize the data
    P{i} = computeP(X{i});
    sumP = sumP + P{i};
end
Wt = computeA(sumP, lambda);

%% convergence processing
Obj_LCCF = zeros(view_num,1);
J_old = 1; J_new = 10; EPS = 1e-3;
iter_num = 0; OFV = [];
while abs((J_new - J_old)/J_old) > EPS
    iter_num = iter_num + 1;
    S = Wt.^lambda;
    DCol = full(sum(S,2));
    D = spdiags(DCol,0,speye(size(S,1)));
    L = D - S;     
    for v = 1 : view_num
        [U{v}, V{v}] = MVCF(X{v}, nClass, Wt, MVCFoptions, U{v}, V{v}, lambda);   
        Obj_LCCF(v) = (norm(X{v} - X{v}*U{v}*V{v}', 'fro'))^2 + 2*trace(V{v}'*L*V{v});
    end 
    
    alpha = EProjSimplex_new(-Obj_LCCF/(2*gamma), 1);
    
    Psum = zeros(point_num);
    for v = 1 : view_num
        P{v} = alpha(v)*computeP(V{v}');
        Psum = Psum + P{v};
    end
    Wt = computeA(Psum, lambda); % update A    
    J_old = J_new;
    J_new = alpha'*Obj_LCCF + gamma*(alpha'*alpha);
end

%% clustering & evaluation
finalV = [];
for vn = 1:view_num
    finalV = [finalV alpha(vn)*V{vn}];
end 


for it = 1:30 % run 30 times to get the best performance
    label = kmeans(finalV,nClass,'EmptyAction','singleton','Start','cluster','Replicates',10);
end
