function varargout = getJthOutput(fun, J, varargin)
     n = nargout(fun);
     outs = cell(1,n);
     [outs{:}] = fun(varargin{:});
     varargout = outs(J);
end
