clear;
clear all;
close
clc

funcprot(0); //Desable the function overwrite warning
 
function [this]=sig(net)//Unipolar sigmoid function; generates o/p and derivative
    this.op=1/(1+exp(-net));
    this.dop=this.op*(1-this.op);
endfunction
function [this]=lrelu(net)//Leaky Rectified Linear Unit function; generates o/p and derivative
    if(net>0)
        this.op=net;
        this.dop=1;
    else
        this.op=.01*net;
        this.dop=.01;
    end
endfunction
function [this]=line(net)//Transfer function for input layer
    this.op=net;
    this.dop=0;
endfunction
function [this]=funct(n)//select  transfer function
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

function [this]=an_int(w,b,fn)// initalise single neuron
    this.w=w;
    this.dw=w*0;
    this.b=b;
    this.fn=fn;
endfunction
function [an]=an_modW(an,dw,b)//neural nework weight modification
    an.w=an.w+dw;
    an.b=an.b+b;
endfunction

function [an]=an_calc(an,x)//calculate output of neuron
    an.x=x;    
    an.net=(x*an.w');//+an.b;
    op=an.fn(an.net);
    an.op=op.op;
    an.dop=op.dop;
endfunction

function [this]=ann_Init(dim,fns) //Initialise neural network
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

function [ann]=ann_forward(ann,x)//ANN forward propagation
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


function [ann]=ann_backward(ann,d,lr,a)//ann Back Propagation
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



function [minmax]=dataset_minmax(dataset)//calculate featurewise minimum and maximum 
    s=size(dataset);
    l=s(2);
    minmax=zeros(l,2);
    for i=1:l
        minmax(i,:)=[min(dataset(:,i)),max(dataset(:,i))];
    end
    return(minmax);
endfunction

function [dataN]=normalize_dataset(dataset)//normalise entire database to 0-1
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
function [ann]=train_ann(ann,trainX,trainY,lr,kfold,maxc,minE,mom)//Train Network
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
            for i=1:s(1)
                ann=ann_forward(ann,trainX(i,:));
                ann=ann_backward(ann,trainY(i,:),lr,mom);
                err=err+(ann.totalE/(s(1)));
            end
            eee=[eee(2:12),err];
            if mee>err
                mee=err;
            end
            disp(mee);
            disp(eee);       
        end        
    end    
endfunction

function [TrX,TrY,TsX,TsY]=train_test_dataSplit2(M)//seperate training and testing dataset
    s=size(M);
    ind=grand(1, "prm", (1:(s(1))));
    M=M(ind,:);
    TrX=M(1:int(.8*s(1)),1:s(2)-1);
    TsX=M(((int(.8*s(1))+1):s(1)),1:s(2)-1);    
    TrY=M(1:int(.8*s(1)),s(2));
    TsY=M(((int(.8*s(1))+1):s(1)),s(2));
endfunction

function [er]=test_ann(ann,testX,testY)// Test ANN with round
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

function [er]=test_ann2(ann,testX,testY)//test ann by compare
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

function [er]=acc(testX,testY)//Accurecy of test results
    s=size(testX);
    er=0;
    for i=1:s(2)
        if(testX(1,i)>testX(2,i))
            a=1
        else
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




function [arr1]=split(arr)//split target matrix for 1 op neuron to 2
    arr1=zeros(length(arr),2);
    arr=arr+1;
    for i=1:length(arr)
        arr1(i,arr(i))=1;
    end
endfunction


////////////////////////////////////////////////////////////////////////////
///////////////////////////Dataset Preperation///////////////////////////////
////////////////////////////////////////////////////////////////////////////


M = csvRead('C:\Users\ktheadmin\Desktop\annassignment\data.csv', ',');//read the database from file
M=normalize_dataset(M);// normalise dataset
[TrX,TrY,TsX,TsY]=train_test_dataSplit2(M);//split training 
TrY=split(TrY)//split 1 neuron  to 2 neuron
TsY=split(TsY)//split 1 neuron  to 2 neuron




////////////////////////////////////////////////////////////////////////////
///////////////////////////Without ANNToolbox///////////////////////////////
////////////////////////////////////////////////////////////////////////////
//save("aaabipol_4.sod","aaa1")

//load("bbb2.sod","bbb")//3 layer 1op neuron
//load("aaabipol_4".sod","bbb")//4 layer 2op neuron
//load("aaa.sod","aaa")//3 layer 2op neuron

 //Initialise Network
//aaa=ann_Init([10,4,2],[3,1,1]);
//train network
//aaa=train_ann(aaa,TrX,TrY,.05,1,10,.06,.2);


//test network
disp(test_ann(aaa,TrX,TrY))

/////////////////////////////////////////////////////////////////////////
////////////////////Using ANN ToolBox////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


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


