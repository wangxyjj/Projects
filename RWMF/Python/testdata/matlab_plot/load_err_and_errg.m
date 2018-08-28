function [x,y,yg] = load_err_and_errg(filename,logflag)

load(filename);
x = 1:length(err_list);
yg = err_g_list;

if logflag == 1
    y = log(err_list);
else
    y = err_list;
end
