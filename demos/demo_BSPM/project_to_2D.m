function X = project_to_2D(Z)
% PROJECT_TO_2D
% 
% Description
%   Mapping from 3D body coordinates to 2D coordinates (using a cylindrical
%   mapping).

    X = zeros(size(Z,1), 2);
    X(:,2) = Z(:,3)/100;
    X(:,1) = atan2(Z(:,1), Z(:,2));
end
