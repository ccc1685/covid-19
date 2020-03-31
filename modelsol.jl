function modelsol(pin,total,N)
	u0 = [0.,0.,0.,1/N,0.,0.]
	p = pin[1:end-1]
	if pin[end] < total - 1
		start = ceil(Int,pin[end])
		tspan = (pin[end],total-1)
		prob = ODEProblem(sicr!,u0,tspan,p)
		sol = solve(prob,saveat=collect(start:total-1))
		# align solutions to time 0 of data and reorder rows and cols
		prediction = zeros(total,3)
		for set in 1:3
			prediction[max(0,start)+1:end,set] .= sol[set,max(0,-start)+1:end]
		end
		return prediction*N, sol
	else
		return zeros(total,3)
	end
end