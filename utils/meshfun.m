function [ f_mesh, grad_mesh, x_coord, y_coord ] = meshfun( fun, x_vec, y_vec, hyp_indizes)

[x_coord,y_coord] = meshgrid(x_vec, y_vec);
f_mesh = zeros(size(x_coord));
grad_mesh = zeros(size(x_coord,1),size(x_coord,2),2);

for i=1:length(x_vec)
    for j=1:length(y_vec)
%         hyp.cov = log([sigma_vec(j); 2*pi; l_vec(i)]);
%         hyp.lik = log(sqrt(sigma2_n));
%         hyp.mean = [];
        try
            [fun_val, grad_fun] = fun(x_vec(i), y_vec(j));
            %grad_fun = [grad_fun.cov; grad_fun.lik];
            %grad_mesh(j,i,:) = grad_fun(hyp_indizes);
        catch e
            %post_dense = [];
            fun_val = nan;
            grad_mesh(j,i,:) = nan;
            fprintf(1,'There was an error! The message was:\n%s',e.message);
        end
        f_mesh(j,i) = fun_val;
        
    end
    disp(['Evaluate Grid: ', num2str(i*length(y_vec)), '/', num2str(length(y_vec)*length(x_vec))])
end

end

