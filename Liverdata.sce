M = csvRead('C:\Users\ktheadmin\Desktop\annassignment\data.csv', ',')
s=size(M)
D1=[];
D2=[];
for i=1:s(1)
    if M(i,11)==1
        D1=[D1,i];
    else
        D2=[D2,i];
    end      
end

D1Tr=D1(1:int(.8*length(D1)));
D2Tr=D2(1:int(.8*length(D2)));
D1Ts=D1((int(.8*length(D1))+1):length(D1));
D2Ts=D2((int(.8*length(D2))+1):length(D2));
Tr=gsort([D1Tr,D2Tr],'g','i');
Ts=gsort([D1Ts,D2Ts],'g','i');
TrX=M(Tr,1:10)
TrY=M(Tr,11)
TsX=M(Ts,1:10)
TsY=M(Ts,11)
TrY=TrY-1;
TsY=TsY-1;
