function [x_projected] = constraints_projection(x, mins, maxs)

x_projected = max(min(x, maxs), mins);

if x_projected == x
    disp('')
else
    disp('projection')
end

end