clear
clc
close all

n = 100;

f = @(x) sum(x(:).^2) - sum(x(1:n-1).*x(2:n));

% Try to change the values characterizing the box
mins = zeros(n,1);
maxs = ones(n,1) * 5;

% check that x0 is in the box
x0 = rand(n,1) * 5;

% system building
kmax = 1000;
tolgrad = 1e-3;
tolx = 1e-6;
projectionf = @(x) constraints_projection(x, mins, maxs);
gamma = 0.5;
alpha0 = 1;
rho = 0.8;
lsmax = 25;
c1 = 1e-4;

for c = 1:4
    
    if c == 1
        disp('Exact derivates computed')
        gradf = @(x) [(2 * x(1) - x(2));2 * x(2:n-1) - x(1:n-2) - x(3:n), ...
         ;2 * x(n) - x(n-1)];
        tic
        [fk, k] = ...
            projected_gradient(x0, f, ...
            gradf, kmax, tolgrad, tolx, projectionf, ...
            gamma, alpha0, c1, rho, lsmax, 0, '')
        time = toc
        f_save = fk;
        time_save = time;
        iters_save = k;
    elseif c == 2
        figure
        disp('Finite difference case fw')
        valk = [];
        fval = [];
        times = [];
        iters = [];
        i = 0;
        for t = 2:2:14
            disp('Value of k : ')
            disp(t)
            gradf = [];
            tic
            [fk, k] = ...
                projected_gradient(x0, f, ...
                gradf, kmax, tolgrad, tolx, projectionf, ...
                gamma, alpha0, c1, rho, lsmax, t, 'fw')
            elapsed_time = toc
            i = i + 1;
            valk(i) = t;
            fval(i) = fk;
            times(i) = elapsed_time;
            iters(i) = k;
            plot(t,f_save, 'b*')
            hold on
            plot([t t], [f_save fk], '--');
        end
        plot(valk,fval,'-om');
        title('Value of the minimum with respect to variation of k, case fw')
        xlabel('K');
        ylabel('Minima');
        saveas(gcf, 'fw1.png')
        figure
        plot(valk, times, '-om')
        hold on
        for t = 2:2:14
            plot(t,time_save, 'b*')
            hold on
            plot([t t], [time_save times(t/2)], '--');
        end
        title('Elapsed time with respect to variation of k, case fw')
        xlabel('K');
        ylabel('Time');
        saveas(gcf, 'fw2.png')
        figure
        plot(valk, iters, '-om')
        hold on
        for t = 2:2:14
            plot(t,iters_save, 'b*')
            hold on
            plot([t t], [iters_save iters(t/2)], '--');
        end
        title('Iterations with respect to variation of k, case fw')
        xlabel('K');
        ylabel('Iterations');
        saveas(gcf, 'fw3.png')
    elseif c == 3
        figure
        valk = [];
        fval = [];
        times = [];
        iters = [];
        i = 0;
        disp('Finite difference case c')
        for t = 2:2:14
            disp('Value of k : ' )
            disp(t)
            gradf = [];
            tic
            [fk, k] = ...
                projected_gradient(x0, f, ...
                gradf, kmax, tolgrad, tolx, projectionf, ...
                gamma, alpha0, c1, rho, lsmax, t, 'c')
            elapsed_time = toc
            i = i + 1;
            valk(i) = t;
            fval(i) = fk;
            times(i) = elapsed_time;
            iters(i) = k;
            plot(t,f_save, 'r*')
            hold on
            plot([t t], [f_save fk], '--');
        end
        plot(valk,fval,'-ob');
        title('Value of the minimum with respect to variation of k, case c')
        xlabel('K');
        ylabel('Minima');
        saveas(gcf, 'c1.png')
        figure
        plot(valk, times, '-ob')
        hold on
        for t = 2:2:14
            plot(t,time_save, 'b*')
            hold on
            plot([t t], [time_save times(t/2)], '--');
        end
        title('Elapsed time with respect to variation of k, case c')
        xlabel('K');
        ylabel('Time');
        saveas(gcf, 'c2.png')
        figure
        plot(valk, iters, '-ob')
        hold on
        for t = 2:2:14
            plot(t,iters_save, 'b*')
            hold on
            plot([t t], [iters_save iters(t/2)], '--');
        end
        title('Iterations with respect to variation of k, case c')
        xlabel('K');
        ylabel('Iterations');
        saveas(gcf, 'c3.png')
    elseif c == 4
        figure
        disp('Finite difference case bw')
        valk = [];
        fval = [];
        times = [];
        iters = [];
        i = 0;
        for t = 2:2:14
            disp('Value of k : ')
            disp(t)
            gradf = [];
            tic
            [fk, k] = ...
                projected_gradient(x0, f, ...
                gradf, kmax, tolgrad, tolx, projectionf, ...
                gamma, alpha0, c1, rho, lsmax, t, 'bw')
            elapsed_time = toc
            i = i + 1;
            valk(i) = t;
            fval(i) = fk;
            times(i) = elapsed_time;
            iters(i) = k;
            plot(t,f_save, 'r*')
            hold on
            plot([t t], [f_save fk], '--');
        end
        plot(valk,fval,'-ok');
        title('Value of the minimum with respect to variation of k, case bw')
        xlabel('K');
        ylabel('Minima');
        saveas(gcf, 'bw1.png')
        figure
        plot(valk, times, '-ok')
        hold on
        for t = 2:2:14
            plot(t,time_save, 'b*')
            hold on
            plot([t t], [time_save times(t/2)], '--');
        end
        title('Elapsed time with respect to variation of k, case bw')
        xlabel('K');
        ylabel('Time');
        saveas(gcf, 'bw2.png')
        figure
        plot(valk, iters, '-ok')
        hold on
        for t = 2:2:14
            plot(t,iters_save, 'b*')
            hold on
            plot([t t], [iters_save iters(t/2)], '--');
        end
        title('Iterations with respect to variation of k, case bw')
        xlabel('K');
        ylabel('Iterations');
        saveas(gcf, 'bw3.png')
    end
end

diary Assignment_2C_results