function procIm = blockproc3(im, blkSize, func)
    if ( ndims(im) ~= length(blkSize) )
        error('Incorrect number of dimensions in blkSize');
    end
    
    % Cheat a bit and apply the function with block of zeros, output should
    % be of "block" output size
    numBlocks = ceil(size(im) ./ blkSize);
    outChk = func(zeros(blkSize));
    outBlkSize = size(outChk);
    if ( length(outBlkSize) < length(blkSize) )
        outBlkSize = [outBlkSize ones(1,length(blkSize)-length(outBlkSize))];
    end
    
    procIm = zeros(outBlkSize.*numBlocks);
    
    blocksI = numBlocks(1);
    blocksJ = numBlocks(2);
    blocksK = numBlocks(3);
    
    inBlockSzI = blkSize(1);
    inBlockSzJ = blkSize(2);
    inBlockSzK = blkSize(3);
    
    outBlockSzI = outBlkSize(1);
    outBlockSzJ = outBlkSize(2);
    outBlockSzK = outBlkSize(3);
    
    for i=1:blocksI
        iIdxStart = (i-1)*outBlockSzI + 1;
        iIdxEnd = i*outBlockSzI;
        
        iInBlockStart = (i-1)*inBlockSzI + 1;
        iInBlockEnd = min(i*inBlockSzI, size(im,1));
        
        for j=1:blocksJ
            jIdxStart = (j-1)*outBlockSzJ + 1;
            jIdxEnd = j*outBlockSzJ;
            
            jInBlockStart = (j-1)*inBlockSzJ + 1;
            jInBlockEnd = min(j*inBlockSzJ, size(im,2));
            
            for k=1:blocksK
                kIdxStart = (k-1)*outBlockSzK + 1;
                kIdxEnd = k*outBlockSzK;

                kInBlockStart = (k-1)*inBlockSzK + 1;
                kInBlockEnd = min(k*inBlockSzK, size(im,3));
                
                procIm(iIdxStart:iIdxEnd,jIdxStart:jIdxEnd,kIdxStart:kIdxEnd) = ...
                    func(im(iInBlockStart:iInBlockEnd,jInBlockStart:jInBlockEnd,kInBlockStart:kInBlockEnd));
            end
        end
    end
end