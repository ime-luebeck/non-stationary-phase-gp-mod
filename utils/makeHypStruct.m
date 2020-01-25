function hyp_struct = makeHypStruct(cov, lik, mean)
    hyp_struct.cov = cov;
    hyp_struct.lik = lik;
    hyp_struct.mean = mean;
end