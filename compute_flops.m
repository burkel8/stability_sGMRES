function numflops = compute_flops(tlog, n)
numflops = 0;
for i = 1:length(tlog)
    numflops = numflops + n*tlog(i);
end
end