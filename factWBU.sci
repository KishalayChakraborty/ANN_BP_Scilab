clear;
clc;

function [o]=fact(net)
    o=1/(1+exp(-net));
endfunction

L=4;
nu=.1;
N=[3,4,2,2]
W=zeros(L,max(N),max(N))
b=zeros(L,max(N))
for l=2:L
    for n=1:N(l)
        W(l,n,1:N(l-1))=rand(1,N(l-1))';
    end
end



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
    for i=1:1:N(L)
        del(L,i)=DY(L,i)*(T(i)'-Y(L,i))
    end
    E=0
    for j=1:N(L)
       E=E+.5*del(L,j)^2
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
XX=[1,2,1;1,1,1;2,2,2]
TT=[1,1;0,1;0,0]
c=0
Err=5
while c<100 & Err>.000005
Err=0
for i=1:3
X=XX(i,:)
T=TT(i,:);
[W,b,E]=pro(W,b,L,nu,N,X,T)
Err=Err+E;
end
disp(Err);
end
for i=1:3
    X=XX(i,:);
    T=TT(i,:)
    N=[3,3,2,2]
[Y]=test(W,b,L,N,X)
disp('test')
disp(T)
disp(Y)
end
