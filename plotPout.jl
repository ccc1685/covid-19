function plotPout(pout)

figure(figsize=(12,6))
for i=1:6
    subplot(2,3,i)
    hist(pout[:,i],bins=100)
    title("p$(i)",fontsize=15)
end
tight_layout()

end