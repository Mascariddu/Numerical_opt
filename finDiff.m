function[fk] = finDiff(f, x, h, type)

fk = zeros(size(x));
n = length(x);

switch type
    case 'fw'
        for i=1:n
%             xh_plus = x;
%             xh_plus(i) = x(i) + h(i);
%             fk(i) = (f(xh_plus) - f(x)) / h(i);
            fk(i) = (f([x(1:i-1); x(i) + h(i); x(i+1:end)]) - f(x))/h(i);
        end
    case 'c'
        for i=1:n
%             xh_plus = x;
%             xh_plus(i) = x(i) + h(i);
%             xh_minus = x;
%             xh_minus(i) = x(i) - h(i);
%             fk(i) = (f(xh_plus) - f(xh_minus)) / 2*h(i);
            fk(i) = (f([x(1:i-1); x(i) + h(i); x(i+1:end)]) - f([x(1:i-1);...
                x(i) - h(i); x(i+1:end)]))/ (2 * h(i));
        end
    case 'bw'
        for i=1:n
%             xh_minus = x;
%             xh_minus(i) = x(i) - h(i);
%             fk(i) = (f(x) - f(xh_minus)) / h(i);
              fk(i) = (f(x) - f([x(1:i-1); x(i) - h(i); x(i+1:end)]))/h(i);
        end
end

end