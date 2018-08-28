function [x,y] = load_coordinates(filename,logflag)

load(filename);
x = 1:length(err_list);

if logflag == 1
    y = log(err_list);
else
    y = err_list;
end
