-- grid search for best set of parameters

function table2str(tbl)
	-- input is {1,2,3,4}
	-- output is "1,2,3,4"
	str = "";
	for i = 1,#tbl do
		str = str .. tostring(tbl[i])
		if i ~= #tbl then
			str = str .. ','
		end
	end
	return str
end

for i = 1,100,10 do
	for j = 1,100,10 do
		for k = 1,100,10 do
			for l = 1,100,10 do
				os.execute("cd ~/othelloAI && ./othello " .. table2str({i,j,k,l}) .. " " .. table2str({1,1,1,1}) .. " " .. " > tmpOutput")
				os.execute("cd ~/othelloAI && echo black " ..  table2str({i,j,k,l}) .. " white " ..  table2str({1,1,1,1}) .. " >> scores")
				os.execute("cd ~/othelloAI && tail -n 2 tmpOutput >> scores")				
			end
		end
	end
end






