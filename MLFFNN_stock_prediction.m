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

%n=input('number of samples for training');
n=100;
%%%%%%%%%%%%%%%%%%%%%%%%%% for identifier %%%%%%%%%%%%%%%%%%%%%%%
w1=0.1*rand(20,1);
w2=0.1*rand(20,1);
w3=0.1*rand(20,1);
w4=0.1*rand(20,1);
w5=0.1*rand(20,1);
w6=0.1*rand(20,1); 
w7=0.1*rand(20,1);
w8=0.1*rand(20,1);
w9=0.1*rand(20,1);
w10=0.1*rand(20,1);
w11=0.1*rand(20,1);%o/p weight matrix
wh=0.1*rand(20,1);
wo=0.1*rand;
%eta=0.0098;
%iter=7000;
%eta=0.009;
eta=0.001;
MSE=0;
iter=1450;  %1450
for i=1:iter
for k=1:100
    
  %%%%%%%%%%%%%%%%%%%%%% for identifier %%%%%%%%%%%%%%%%%%%
    net1=tansig(x1(k)*w1+x2(k)*w2+x3(k)*w3+x4(k)*w4+x5(k)*w5+x6(k)*w6+x7(k)*w7+x8(k)*w8+x9(k)*w9+x10(k)*w10-wh); 
    y(k)=purelin(net1'*w11-wo);
    e(k)=yd(k)-y(k);
    MSE=MSE+0.5*e(k)^2;
  a=dpurelin(net1'*w11-wo,y(k))*e(k); %gradient of o/p. its a scalar
    b=dtansig(x1(k)*w1+x2(k)*w2+x3(k)*w3+x4(k)*w4+x5(k)*w5+x6(k)*w6+x7(k)*w7+x8(k)*w8+x9(k)*w9+x10(k)*w10-wh,net1); %derivative of hidden neuron AF, 20x1 vector
    delw1=eta*x1(k)*a*b.*w11;
    delw2=eta*x2(k)*a*b.*w11;
    delw3=eta*x3(k)*a*b.*w11;
     delw4=eta*x4(k)*a*b.*w11;
      delw5=eta*x5(k)*a*b.*w11;
      delw6=eta*x6(k)*a*b.*w11;
      delw7=eta*x7(k)*a*b.*w11;
      delw8=eta*x8(k)*a*b.*w11;
      delw9=eta*x9(k)*a*b.*w11;
      delw10=eta*x10(k)*a*b.*w11;
    delw11=eta*net1*a;
    delwh=eta*(-1)*a*b.*w11;
    delwo=eta*(-1)*a;
    w1=w1+delw1;
    w2=w2+delw2;
    w3=w3+delw3;
    w4=w4+delw4;
    w5=w5+delw5;
    w6=w6+delw6;
    w7=w7+delw7;
    w8=w8+delw8;
    w9=w9+delw9;
    w10=w10+delw10;
    w11=w11+delw11;
    wo=wo+delwo;
    wh=wh+delwh;
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