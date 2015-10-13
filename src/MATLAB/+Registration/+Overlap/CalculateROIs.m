function [start1,start2,end1,end2] = CalculateROIs(delta,oldStart1,oldStart2,size1,size2)
if (oldStart1==1 && oldStart2~=1)
    start1 = 1;
else
    start1 = max(1, oldStart1+delta);
end

if (oldStart2==1 && oldStart1~=1)
    start2 = 1;
else
    start2 = max(1, oldStart2-delta);
end

minSize = min(size1-start1,size2-start2);
end1 = start1 + minSize;
end2 = start2 + minSize;

if (end1-start1~=end2-start2),error('Sizes dont`t match %d : %d!',end1-start1,end2-start2), end
end
