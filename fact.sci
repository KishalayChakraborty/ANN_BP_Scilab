clear;
clc;

function [o]=fact(net)
    o=1/(1+exp(-net));
endfunction

function [YY]=test(W,b,L,N,X)
    
   Y1(1,:)=X

    for l=2:L
        for n=1:N(l)
         x=W(l,n,1:N(l-1))//'*
           s=Y1(l-1,1:N(l-1))
           a=0;//x'*s
             for(i=1:length(s))
                a=a+(x(:,:,i)*s(i));
             end        
             Y1(l,n)=fact((a+b(l,n)));            
        end
        //disp(N)
        //
    end
    YY=Y1(L,1:N(L))
    
endfunction
function [W,b,E]=pro(W,b,L,nu,N,X,T)
    
    Y(1,:)=X
    for i=1:length(X)    
        DY(1,i)=X(i)*(1-X(i));
    end
    for l=2:L
        for n=1:N(l)
            x=W(l,n,1:N(l-1))//'*
            s=Y(l-1,1:N(l-1))
            a=0;//x'*s
            for(i=1:length(s))
                a=a+(x(:,:,i)*s(i));
            end        
            Y(l,n)=fact((a+b(l,n)));
            DY(l,n)=Y(l,n)*(1-Y(l,n))
        end
    end
    
    //7done
    del=zeros(L,max(N));
    for i=1:N(L)
        del(L,i)=DY(L,i)*(T(i)'-Y(L,i))
    end
    E=0
    for j=1:N(L)
       E=E+.5*(T(j)'-Y(L,j))^2
    end
    for ll=1:(L-1)
        l=L-ll
        for n=1:N(l+1)
            a=0;//x'*s
            for(i=1:N(l))
                a=a+(W(l+1,n,i)*del((l+1),i))
            end 
            del(l,n)=a*DY(l,n)
        end    
    end
    
    delW=zeros(L,max(N),max(N));
    delB=zeros(L,max(N));
    for ll=0:L-2
        l=L-ll
        for n=1:N(l)
            delW(l,n,1:N(l-1))=nu*(del(l,n)*Y(l-1,1:N(l-1)));
            b(l,n)=b(l,n)+del(l,n)  ;
            W(l,n,1:N(l-1))=W(l,n,1:N(l-1))+ delW(l,n,1:N(l-1))  ;
               
        end    
    end
endfunction





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
for i=1:s(2)-1
    M(1:s(1),i)=(max( M(1:s(1),i))- M(1:s(1),i))/(max( M(1:s(1),i))-min( M(1:s(1),i)))
     
end
D1Tr=D1(1:int(.8*length(D1)));
D2Tr=D2(1:int(.8*length(D2)));
D1Ts=D1((int(.8*length(D1))+1):length(D1));
D2Ts=D2((int(.8*length(D2))+1):length(D2));
Tr=gsort([D1Tr,D2Tr],'g','i');
Ts=gsort([D1Ts,D2Ts],'g','i');
TrX=M(Tr,1:10)
TsX=M(Ts,1:10)

TrY=zeros(length(Tr),2);//M(Tr,11)
TsY=zeros(length(Ts),2);//M(Ts,11)
for i=1:length(Tr)
    TrY(i,M(Tr(i),11))=1;
end
for i=1:length(Ts)
    TsY(i,M(Ts(i),11))=1;
end







L=3;
nu=.5;
N=[10,4,2]

Dc=465
XX=TrX;//[1,2,1;1,1,1;2,2,2;1,2,5]
TT=TrY;//[1,1;0,1;0,0;1,0]
W=zeros(L,max(N),max(N))
b=zeros(L,max(N))
for l=2:L
    for n=1:N(l)
        W(l,n,1:N(l-1))=rand(1,N(l-1))';
    end
end

c=0
Err=5
while c<10000000 & Err>.15
    Err=0;
    for i=1:Dc
        //i=1
        X=XX(i,1:10);
        T=TT(i,:);
        [W,b,E]=pro(W,b,L,nu,N,X,T);
        Err=Err+E;
    end
    Err=Err/Dc
    disp(Err);
    disp(c);
    c=c+1;
end

XX=TrX;//[1,2,1;1,1,1;2,2,2;1,2,5]
TT=TrY;
Dc=465
EEE=[]
for i=1:Dc
    X=XX(i,1:10);
    T=TT(i,:);
    //N=[3,3,2,2];
    [Y]=test(W,b,L,N,X)
    disp('test')
    disp(T)
    disp(Y)
    bb=floor(Y(1)-Y(2))
    if(bb<0)
        Y=[0,1]    
    else
        Y=[1,0]
    end
    
       EEE=[EEE,sum(abs(T-Y))/2];
    
end
disp(100*sum(EEE)/Dc)
