clear;
clear all;
close
clc
funcprot(0)
function [this]=sig(net)
    //disp(net)
    this.op=1/(1+exp(-net));
    this.dop=this.op*(1-this.op);
endfunction
function [this]=lrelu(net)
    if(net>0)
        this.op=net;
        this.dop=1;
    else
        this.op=.01*net;
        this.dop=.01;
    end
    
endfunction
function [this]=line(net)
    this.op=net;
    this.dop=0;
endfunction
function [this]=funct(n)
    if n==1 then
        this=sig;
    elseif n==2 then
        this=line
    elseif n==3 then
        this=lrelu
    else
        
        this=sig;
    end
endfunction

function [this]=an_int(w,b,fn)
    this.w=w;
    this.dw=w*0;
    this.b=b;
    this.fn=fn;
endfunction
function [an]=an_modW(an,dw,b)
    //disp(an.w)
    //disp(dw)
    an.w=an.w+dw;
    an.b=an.b+b;
endfunction

function [an]=an_calc(an,x)
    an.x=x;
    //disp(x);
    //disp(an.w);
    
    an.net=(x*an.w');//+an.b;
    op=an.fn(an.net);
    an.op=op.op;
    an.dop=op.dop;
endfunction

function [this]=ann_Init(dim,fns)
    this.dim=dim;
    ann=[];
    for(j=1:dim(1))
        w=rand(1,dim(1))
        this.layer(1).an(j)=[an_int(w,0,funct(fns(1)))];
    end   
    for i=2:length(dim)
        for(j=1:dim(i))
            this.layer(i).an(j)=[an_int(((rand(1,dim(i-1)))),((rand(1,1)*2)-1),funct(fns(i)))];           
        end
    end
endfunction

function [ann]=ann_forward(ann,x)
    dim=ann.dim;
    for(j=1:dim(1))
        ann.layer(1).an(j).op=x(j);//[an_calc(ann.layer(1).an(j),x)];
    end   
    for i=2:length(dim)
        for(j=1:dim(i))
            y=[];
            for k=1:length(ann.layer(i-1).an(:).op)
                y=[y,ann.layer(i-1).an(:).op(k)];
            end
            ann.layer(i).an(j)=[an_calc(ann.layer(i).an(j),y)];           
        end
    end
endfunction


function [ann]=ann_backward(ann,d,lr,a)
    dim=ann.dim;
    totalE=0;
    for(j=1:dim(length(dim)))
        ann.layer(length(dim)).an(j).del=-((d(j)-ann.layer(length(dim)).an(j).op))*ann.layer(length(dim)).an(j).dop;        
        ann.layer(length(dim)).an(j).err=.5*(d(j)-ann.layer(length(dim)).an(j).op)^2;
        totalE=totalE+ann.layer(length(dim)).an(j).err;
        
    end     
    ann.totalE=totalE/dim(length(dim));
    for c=1:length(dim)-2
        i=length(dim)-c;
        for(j=1:dim(i))
            def=0;
            for k=1:dim(i+1)
                def=def+(ann.layer(i+1).an(k).w(j)*ann.layer(i+1).an(k).del);
            end
            //disp(ann.layer(i).an(j))
            ann.layer(i).an(j).del=def*ann.layer(i).an(j).dop;
           
        end
    end
    for c=0:length(dim)-2
        i=length(dim)-c;
        for(j=1:dim(i))
            ann.layer(i).an(j).dw=(-lr*ann.layer(i).an(j).del*ann.layer(i).an(j).x)*(1-a)+ann.layer(i).an(j).dw*a;
            ann.layer(i).an(j).w=ann.layer(i).an(j).w+ann.layer(i).an(j).dw;
            ann.layer(i).an(j).b=ann.layer(i).an(j).b-ann.layer(i).an(j).del*lr; 
    
           
        end
    end

endfunction



function [minmax]=dataset_minmax(dataset)
    s=size(dataset);
    l=s(2);
    minmax=zeros(l,2);
    for i=1:l
        minmax(i,:)=[min(dataset(:,i)),max(dataset(:,i))];
    end
    return(minmax);
endfunction


function [dataN]=normalize_dataset(dataset)
    minmax=dataset_minmax(dataset);
    s=size(dataset);
    l=s(1);
    n=s(2);
    dataN=zeros(l,n);
    for i=1:n
        dataN(:,i)=(dataset(:,i)-minmax(i,1))/ (minmax(i,2)-minmax(i,1));      
    end
    return(dataN);
endfunction



function [ann]=train_ann(ann,trainX,trainY,lr,kfold,maxc,minE,mom)
    s=size(trainX);
    cnt=0;
    err=1000;
    mee=1000;
    eee=[0,0,0,0,0,0,0,0,0,0,0,0]
    while cnt<maxc & err>minE
        
        for a=1:kfold
            err=0
            disp(cnt);
            cnt=cnt+1;
            ind=grand(1, "prm", (1:(s(1))));
            trainX=trainX(ind,:);
            trainY=trainY(ind,:);
            //disp(s(1))
            for i=1:s(1)
                ann=ann_forward(ann,trainX(i,:));
                //err=err+ann.totalE;
                ann=ann_backward(ann,trainY(i,:),lr,mom);
                err=err+(ann.totalE/(s(1)));
                //disp(err);
            end
            

            eee=[eee(2:12),err]//err=err/kfold;
            if mee>err
                mee=err;
            end
            disp(mee);
            disp(eee);
        
        end        

    end    
endfunction

function [TrX,TrY,TsX,TsY]=train_test_dataSplit2(M)
    s=size(M);
    ind=grand(1, "prm", (1:(s(1))));
    M=M(ind,:);
    TrX=M(1:int(.8*s(1)),1:s(2)-1);
    TsX=M(((int(.8*s(1))+1):s(1)),1:s(2)-1);    
    TrY=M(1:int(.8*s(1)),s(2));
    TsY=M(((int(.8*s(1))+1):s(1)),s(2));
endfunction
function [TrX,TrY,TsX,TsY]=train_test_dataSplit(M)
    s=size(M);
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
    TrX=M(Tr,1:s(2)-1);
    TsX=M(Ts,1:s(2)-1);    
    TrY=M(Tr,s(2));//zeros(length(Tr),2);//M(Tr,11)
    TsY=M(Ts,s(2));//zeros(length(Ts),2);//M(Ts,11)
    //for i=1:length(Tr)
    //    TrY(i,M(Tr(i),11))=1;
    //end
    //for i=1:length(Ts)
    //    TsY(i,M(Ts(i),11))=1;
    //end
endfunction


function [er]=test_ann(ann,testX,testY)
    s=size(testX);
    er=0;
    for i=1:s(1)
        ann=ann_forward(ann,testX(i,:));
        dim=ann.dim;
        for j=1:dim(length(dim))
            
            er=er+abs(round(ann.layer(length(dim)).an(j).op)-testY(i,j));
            disp(ann.layer(length(dim)).an(j).op)
            disp(testY(i,j))
        end
    end
    er=100*er/(dim(length(dim))*s(1));
    
endfunction

function [er]=test_ann2(ann,testX,testY)
    s=size(testX);
    er=0;
    for i=1:s(1)
        ann=ann_forward(ann,testX(i,:));
        dim=ann.dim;
        if(ann.layer(length(dim)).an(1).op>ann.layer(length(dim)).an(2).op)
            a=1
        elseif(ann.layer(length(dim)).an(1).op<=ann.layer(length(dim)).an(2).op)
            a=2
        end
        if(testY(i,a)==1)
            c=0
        else
            c=1
        end
        er=er+c;
    end
    er=100*er/s(1);
    
endfunction



function [er]=acc(testX,testY)
    s=size(testX);
    er=0;
    for i=1:s(2)
        if(testX(1,i)>testX(2,i))
            a=1
        else//if(ann.layer(length(dim)).an(1).op<=ann.layer(length(dim)).an(2).op)
            a=2
        end
        if(testY(a,i)==1)
            c=0
        else
            c=1
        end
        er=er+c;
    end
    er=100*er/s(2);
    
endfunction




function [arr1]=split(arr)
    arr1=zeros(length(arr),2);
    arr=arr+1;
    for i=1:length(arr)
        arr1(i,arr(i))=1;
    end
endfunction


//save("aaabipol_4.sod","aaa1")
//load("aaa.sod","aaa")

M = csvRead('C:\Users\ktheadmin\Desktop\annassignment\data.csv', ',');
M=normalize_dataset(M);
[TrX,TrY,TsX,TsY]=train_test_dataSplit2(M);
TrY=split(TrY)
TsY=split(TsY)

//TrX=[0,0;0,1;1,0;1,1];  
//TrY=[1;0;0;1];  
aaa1=ann_Init([10,4,3,2],[3,1,1,1]);
aaa1=train_ann(aaa1,TrX,TrY,.05,1,10,.06,.2);





aaa1=train_ann(aaa1,TrX,TrY,.2,1,100,.06,.1);
aaa=train_ann(aaa,TrX,TrY,.5,1,15,.06,.4);
aaa=train_ann(aaa,TrX,TrY,.2,1,15,.06,.2);
aaa=train_ann(aaa,TrX,TrY,.1,1,20,.06,.4);
aaa=train_ann(aaa,TrX,TrY,.01,1,50,.06,.5);
aaa=train_ann(aaa,TrX,TrY,.01,1,50,.06,.5);
aaa=train_ann(aaa,TrX,TrY,.01,1,500,.06,.2);



disp(test_ann2(aaa,TrX,TrY))
aaa=train_ann(aaa,TrX,TrY,.8,1,10,.06,.4)

disp(test_ann(bbb,TrX,TrY))
aaa=train_ann(aaa,TrX,TrY,.5,1,10,.06,.3)
aaa=train_ann(aaa,TrX,TrY,.1,1,15,.06,.4)





aaa=train_ann(aaa,TrX,TrY,.02,1,5,.06,.7)
disp(test_ann(aaa,TrX,TrY))


//defination of layers
N = [10 4 2];
W = ann_FF_init(N);
x=TrX';
t=TrY';
lp = [0.01, 1e-4];
epochs = 300;
W = ann_FF_Std_batch(x,t,N,W,lp,epochs);
y = ann_FF_run(x,N,W)
disp(y)


