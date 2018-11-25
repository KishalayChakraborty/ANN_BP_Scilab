//clear;
//clear all;
close
clc
function [this]=an_sig_int(w,b,l)
    this.w=w;
    this.b=b;
    this.l=l;
endfunction
function [an]=an_sig_modW(an,dw,b)
    //disp(an.w)
    //disp(dw)
    an.w=an.w+dw;
    an.b=b;
endfunction
function [an]=an_sig_up_cal(an,x)
    an.ip=x;
    net=0;
    for i=1:length(x)
        net=net+x(i)*an.w(i)
    end
    an.net=net;
    an.op=1/(1+exp(-net*an.l));
    an.dop=an.op*(1-an.op)
endfunction

function [this]=ann_init(ip,h,op)
    lyr=[];
    this.dim=[ip,h,op]
    
    lyr=[an_sig_int((rand(1,ip)*2),0,1)];
    for i=2:h
        x=an_sig_int((rand(1,ip)*2),0,1);
        lyr=[lyr,x]
    end
    this.h=lyr;
    lyr=[an_sig_int((rand(1,ip)*2),0,1)];
    for i=2:op
        lyr=[lyr,an_sig_int((rand(1,ip)*2),0,1)]
    end
    this.op=lyr;
endfunction

function [ann]=ann_for(X,ann)
    ipn=ann.dim(1);
    hn=ann.dim(2);
    opn=ann.dim(3);    
    ann.ip=X;        
    op2=[];
    for (i=1:hn)
        ann.h(i)=an_sig_up_cal(ann.h(i),ann.ip);
        op2=[op2,ann.h(i).op]
    end    
    for i=1:opn
        ann.op(i)=an_sig_up_cal(ann.op(i),op2);
    end
endfunction

function [ann]=ann_backprop(D,ann,nu)
    ipn=ann.dim(1);
    hn=ann.dim(2);
    opn=ann.dim(3);
    
    del1=[];
    w1=[]
    for i=1:opn
        d=ann.op(i).dop*(D(i)-ann.op(i).op)
        del1=[del1,d];
        w1=[w1,ann.op(i).w']
    end
    
    del2=[];
    for i=1:hn
        ss=0
        for k=1:opn
            ss=ss+(del1(k)*w1(i,k))
        end        
        d=(ss)*(ann.h(i).dop)
        del2=[del2,d];
    end    
    
    for i=1:opn
        w=ann.op(i).w;
        dw=nu*(ann.op(i).ip).*del1;
        ann.op(i)=an_sig_modW(ann.op(i),dw,0)
    end 
    for i=1:hn
        w=ann.h(i).w;
        dw=nu*(ann.h(i).ip).*del2;
        ann.h(i)=an_sig_modW(ann.h(i),dw,0)
    end
    
endfunction


function [e]=ann_err_cal(d,ann)
    e=0
    opn=ann.dim(3);
    for i=1:opn
        e=e+(.5*((d(i)-ann.op(i).op)^2));
    end
    return e;
endfunction
function [ann1]=ann_train(Z,d,dim,nu,Emax)
    s=size(Z);
    dim(1)=s(2);
    s=s(1);
    c=0
    E=Emax+1;
    ann1=ann_init(dim(1),dim(2),dim(3))
    Emin=100;
    while((E>Emax) & c<10000)
        E=0;
        c=c+1
        for i=1:s
            X=Z(i,:)
            ann1=ann_for(X,ann1);
            E=E+ann_err_cal(d(i,:),ann1);
            ann1=ann_backprop(d(i,:),ann1,nu);
        end

        //nu=nu*(E-Emax);
        
    disp('err')
    disp(E)
    
    if E<Emin
        Emin=E
    end
    
end

    disp('minE')
disp(Emin)
    
    disp('cnt')
    disp(c)
endfunction
function [ann]=ann_test(Z,ann)
    s=size(Z);
    dim(1)=s(2);
    s=s(1);
    for i=1:s
            X=Z(i,:)
            ann=ann_for(X,ann);
            disp(X)
           for j=1:1
                disp(ann.op(j).op)
           end
    end
endfunction



z=[0,0,0;0,0,1;1,0,0;1,1,1]
d=[1,1;0,0;0,1;1,0]
ann2=ann_train(z,d,[3,3,2],1,.35)
ann_test(z,ann2)
