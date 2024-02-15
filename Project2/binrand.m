function u = binrand(T,N,tmin,t0,dist)

Width = (length(T)-tmin*N-t0);

m = 0:2*Width/N;

if(strcmp(dist,'uniform'))
    Prob = ones(size(m));
elseif(strcmp(dist,'normal'))
    Prob = normpdf(m,Width/N,(Width/N)/4);
elseif(strcmp(dist,'exponential'))
    Prob = exp(-(N/Width)*m);
else
    disp('Error For Dist')
end

j = 1;
while j < 2000
    amp = zeros(N-1,1);
    for i = 1:N-1
        x = cumsum(Prob)/sum(Prob)
        [row,col,v] = find(x>rand(),1);
        amp(j)=m(col);
        if(amp(j)<0)
            amp(j)=0;
        end
    end
    if(sum(amp)<Width)
        break;
    end
    j=j+1;
end

puls = amp+tmin;
puls(N)= Width-sum(amp)+tmin;
Q = 0;
u = zeros(1,t0);
levels = -1:1;

for i=1:N
    choices = levels;
    choices(Q+2)=[];
    Q = choices(randi([1 2]));
    u=[u Q*ones(1,puls(i))];
end
end
