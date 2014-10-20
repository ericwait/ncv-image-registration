nodes1 = zeros(1,length(edges));
nodes2 = zeros(1,length(edges));
W = zeros(1,length(edges));

for i=1:length(edges)
    if (~isempty(edges(i).nodeIdx1))
        nodes1(i)=edges(i).nodeIdx1;
        nodes2(i)=edges(i).nodeIdx2;
        W(i)=edges(i).normCovar;
    end
end