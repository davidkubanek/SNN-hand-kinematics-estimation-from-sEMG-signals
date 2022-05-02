function [ CH, coIDX, approximationRank_tutto, explained, COtutto, PItutto] = fun_getPC_channel(obs, cha_med, varianza)

% obs = obs1;

condizione = true;

while  (condizione == true)
    
    [~,   score, ~, ~, explained, ~] = pca(obs(:,cha_med), 'Rows','pairwise');
    
    [~, aaa] = find ( (cumsum(explained) - varianza ) > 0 , 1) ;
    IDX_N_PC = 1:aaa; %(find(cumsum(explained) < varianza ));
    
    co = []; pi = [];
 
    for i = cha_med
        [pi(i), co(i) ] = corr(score(:,1), obs(:,i));
    end
    
%     figure, stem(co.*(pi<0.05))
    [~, pos_corr_max] = find(abs(co)>0.95);
    
    if ( (~isempty( pos_corr_max)) &&  (length(pos_corr_max)<2) )
        
        cha_med(find(ismember(cha_med, pos_corr_max))) =  [];
        continue
        
    else
        
        approximationRank_tutto = score(:,IDX_N_PC) ; %* coef(:,1:IDX_N_PC)'; %+ repmat(mu, 4170, 1);end
        condizione = false;
        
    end
end

%  IDX_N_PC = [IDX_N_PC;  IDX_N_PC(end)+1];

if length(IDX_N_PC) < 3
    IDX_N_PC = [IDX_N_PC; 3];
end

clear co pi CH CORR
pi = zeros(1,56);  co = pi; CH = zeros(1, IDX_N_PC(end) );
 
PItutto = ones(56,length(cha_med));
COtutto = zeros(56,length(cha_med));

for ix = 1:length(score(1,:))  %% verificare tutto
    for i = cha_med         % check solo su cha_med così quelli esclusi restano a zero
        [pi(i), co(i) ] = corr(score(:,ix), obs(:,i));
         [COtutto(i, ix), PItutto(i, ix) ] = corr(score(:,ix), obs(:,i));
    end
    
   
    
     co = abs(co);
     vv = (co).*(pi<0.05) ;    % figure, stem((co).*(pi<0.05))

     if sum(vv)>0
        [CORR(ix) , CH(ix)] = max((co).*(pi<0.05)  );
     end
    
end


[~, IDX] = sort(CORR, 'descend');
CH = CH(IDX);
coIDX = CORR(IDX);
CH = CH(CH>0);
coIDX = coIDX(coIDX>0);

[CH, chi] = unique(CH, 'stable');
coIDX = coIDX(chi);



end

