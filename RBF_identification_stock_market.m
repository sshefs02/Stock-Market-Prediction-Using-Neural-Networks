%%%%%%%%%%%%%%%%% RBF BASED IDENTIFICATION USING GRADIENT DESCENT%%%%%%%%%%
%since the range of some inputs were very large so as of o/p, they have
%been normalized using max-min formula..the similar results are obtained
%if normalization is done by dividing each signal w.r.t its max value.
clc
clear all
%close all
load individual_inputs
x1=(x1-min(x1))./(max(x1)-min(x1));
x2=(x2-min(x2))./(max(x2)-min(x2));
x3=(x3-min(x3))./(max(x3)-min(x3));
x4=(x4-min(x4))./(max(x4)-min(x4));
x5=(x5-min(x5))./(max(x5)-min(x5));
x6=(x6-min(x6))./(max(x6)-min(x6));
x7=(x7-min(x7))./(max(x7)-min(x7));
x8=(x8-min(x8))./(max(x8)-min(x8));
x9=(x9-min(x9))./(max(x9)-min(x9));
x10=(x10-min(x10))./(max(x10)-min(x10));
load stock_output
yd=stock_output(:,1);
yd=(yd-min(yd))./(max(yd)-min(yd));
%p=30;
p=30;
%n=input('number of samples for training');
n=100;
h=2*rand(p,10)-1;
I=ones(p,1);
w1=rand(p,1);
rou=2/sqrt(2*p);
%eta=0.0098;
%iter=7000;
%eta=0.009;
eta=0.0099;
MSE=0;
iter=250; %1450
for i=1:iter
for k=1:100
    
    arg=((x1(k)*I-h(:,1)).^2+(x2(k)*I-h(:,2)).^2+(x3(k)*I-h(:,3)).^2+(x4(k)*I-h(:,4)).^2+(x5(k)*I-h(:,5)).^2+(x6(k)*I-h(:,6)).^2+(x7(k)*I-h(:,7)).^2+(x8(k)*I-h(:,8)).^2+(x9(k)*I-h(:,9)).^2+(x10(k)*I-h(:,10)).^2);
    phi=exp(-arg./2*rou^2);
    y(k)=phi'*w1;
    e(k)=yd(k)-y(k);
    MSE=MSE+0.5*e(k)^2;
    h(:,1)=h(:,1)+eta*e(k).*w1.*phi.*(x1(k)*I-h(:,1))./rou^2;
    h(:,2)=h(:,2)+eta*e(k).*w1.*phi.*(x2(k)*I-h(:,2))./rou^2;
    h(:,3)=h(:,3)+eta*e(k).*w1.*phi.*(x3(k)*I-h(:,3))./rou^2;
    h(:,4)=h(:,4)+eta*e(k).*w1.*phi.*(x4(k)*I-h(:,4))./rou^2;
    h(:,5)=h(:,5)+eta*e(k).*w1.*phi.*(x5(k)*I-h(:,5))./rou^2;
    h(:,6)=h(:,6)+eta*e(k).*w1.*phi.*(x6(k)*I-h(:,6))./rou^2;
    h(:,7)=h(:,7)+eta*e(k).*w1.*phi.*(x7(k)*I-h(:,7))./rou^2;
    h(:,8)=h(:,8)+eta*e(k).*w1.*phi.*(x8(k)*I-h(:,8))./rou^2;
    h(:,9)=h(:,9)+eta*e(k).*w1.*phi.*(x9(k)*I-h(:,9))./rou^2;
    h(:,10)=h(:,10)+eta*e(k).*w1.*phi.*(x10(k)*I-h(:,10))./rou^2;
    w1=w1+eta*e(k)*phi;
end
AMSE(i)=MSE/k;
MSE=0;
end

figure
plot(yd,'g')
hold on
plot(y,'r:')
legend('Response of a plant', 'Identification model output')
xlabel('Instants')
ylabel('y_p and y_R')
figure
plot(AMSE)
xlabel('epochs')
ylabel('Average mean square error')
legend('MSE of RBFN')