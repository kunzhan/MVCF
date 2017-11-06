function A = computeA(P, lambda)
    c = size(P,1);
    Pt = zeros(c);
    xp = 1/(1 - lambda);    
    Pt(P~=0) = P(P~=0).^xp;    
    rs = sum(Pt,2); 
    A = bsxfun(@rdivide, Pt, rs);
end
