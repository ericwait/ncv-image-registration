function box = getBox(curCenter, orgImageSize, reduction, nextWindowSize)    

curCenter = round(center/reduction);

box(1) = max(1,curCenter(2)-nextWindowSize(1)/2);
box(4) = min(orgImageSize(1),curCenter(2)+nextWindowSize(1)/2);
box(2) = max(1,curCenter(1)-nextWindowSize(2)/2);
box(3) = min(orgImageSize(2),curCenter(1)+nextWindowSize(2)/2);


lowXstartIdx = centerXorg-xRadius*factorLowRes;
lowXendIdx = centerXorg+xRadius*factorLowRes;
lowYstartIdx = centerYorg-yRadius*factorLowRes;
lowYendIdx = centerYorg+yRadius*factorLowRes;

if (lowXendIdx>imOrgSizes(2))
    lowXendIdx = imOrgSizes(2);
    lowXstartIdx = max(1,imOrgSizes(2)-xRadius*2*factorLowRes);
elseif (lowXstartIdx<1)
    lowStartIdx = 1;
    lowXendIdx = min(imOrgSizes(2),1+xRadius*2*factorLowRes);
end
if (lowYendIdx>imOrgSizes(1))
    lowYendIdx = imOrgSizes(1);
    lowYstartIdx = max(1,imOrgSizes(1)-yRadius*2*factorLowRes);
elseif (lowYstartIdx<1)
    lowYstartIdx = 1;
    lowYendIdx = min(imOrgSizes(1),1+yRadius*2*factorLowRes);
end

highXstartIdx = centerXorg-xRadius*factorHighRes;
highXendIdx = centerXorg+xRadius*factorHighRes;
highYstartIdx = centerYorg-yRadius*factorHighRes;
highYendIdx = centerYorg+yRadius*factorHighRes;

if (highXendIdx>imOrgSizes(2))
    highXendIdx = imOrgSizes(2);
    highXstartIdx = max(1,imOrgSizes(2)-xRadius*2*factorHighRes);
elseif (highXstartIdx<1)
    highXstartIdx = 1;
    highXendIdx = min(imOrgSizes(2),1+xRadius*2*factorHighRes);
end
if (highYendIdx>imOrgSizes(1))
    highYendIdx = imOrgSizes(1);
    highYstartIdx = max(1,imOrgSizes(1)-yRadius*2*factorHighRes);
elseif (highYstartIdx<1)
    highYstartIdx = 1;
    highYendIdx = min(imOrgSizes(1),yRadius*2*factorHighRes);
end