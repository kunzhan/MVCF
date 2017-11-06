function P = computeP(M)   
    c = size(M,2);
    P = zeros(c);
    for i = 1:c
        for j = 1:c
            temp = M(:,i) - M(:,j);
            P(i,j) = temp'*temp;
        end
    end
    Small = min(P, 2);
    for k = 1:c
        P(k,k) = Small(k)/10;
    end
end
