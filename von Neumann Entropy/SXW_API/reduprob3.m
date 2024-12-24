%%%%%%% Programmed by Xiao-Wen Shang 
%%%%%%% 2024 12 21

function s = reduprob3(U) % s是4*4矩阵
s = zeros(4,4,4);
N_mod = size(U,1);
N_col = (N_mod-6)/2;
k = [1,2];
l = [N_col+3,N_col+4];
m = [N_mod-1,N_mod];
B = zeros(2,2,4);
B(:,:,1) = eye(2);
B(:,:,2) = 1/sqrt(2)*[1,1;1,-1];
B(:,:,3) = 1/sqrt(2)*[1,-1i;1,1i];
B(:,:,4) = eye(2);
E = [1,1,1,1; 1,-1,-1,-1];
e = zeros(2,2,2);
Ut = eye(N_mod); % quantum tomography

for index4 = 1 : 4 % 粒子1
    for index5 = 1 : 4 % 粒子2
        for index6 = 1 : 4 % 粒子3
            Ut(end-1:end,end-1:end) = B(:,:,index4);
            Ut(N_col+3:N_col+4,N_col+3:N_col+4) = B(:,:,index5);
            Ut(1:2,1:2) = B(:,:,index6);
            Ur = (Ut*U)';
            p = zeros(2,2,2);
            for index1 = 1 : 2 % 粒子3
                for index2 = 1 : 2 % 粒子2
                    for index3 = 1 : 2 % 粒子1
                        p1 = perm3(Ur([2,N_col+3,end-1],[k(index1),l(index2),m(index3)]));
                        p2 = perm3(Ur([2,N_col+4,end],[k(index1),l(index2),m(index3)]));
                        p(index3,index2,index1) = 0.5*norm(p1)^2 + 0.5*norm(p2)^2 + real(p1*p2');
                    end
                end
            end
            e(:,:,1) = [E(1,index4)*E(1,index5), E(1,index4)*E(2,index5); ...
                E(2,index4)*E(1,index5), E(2,index4)*E(2,index5)]*E(1,index6);
            e(:,:,2) = [E(1,index4)*E(1,index5), E(1,index4)*E(2,index5); ...
                E(2,index4)*E(1,index5), E(2,index4)*E(2,index5)]*E(2,index6);
            s(index4,index5,index6) = sum(sum(sum(e.*p))); % fprintf('%f\n', sum(sum(sum(p))));
        end
    end
end
% s = s/sum(sum(sum(p))); % 确定是否归一化
% fprintf('%f\n', sum(sum(sum(p))));
% s(1,1,1) = sum(sum(sum(p)));
s(1,1,1) = 1;
end

