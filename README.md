# AtkSE
Pytorch implementation for the gray-box structural attacker AtkSE from CIKM22 paper 'Are Gradients on Graph Structure Reliable in Gray-box Attacks?'

In this work, we are concerned about the errors in the gradient from backpropagation. Inspired by saliency methods, we implement ways to eliminate the errors so we have more reliable saliency for the attackers. The only drawback is that we use some time-consuming ensemble algorithms.

The Python file <demo.py> is used to generate a perturbed graph. The <test.py> is used to test the perturbed graphs on the victim model under poisoning attack scenarios.

I strongly recommend our NeurIPS 2022 paper 'Towards Reasonable Budget Allocation in Untargeted Graph Structure Attacks via Gradient Debias' to audiences who are interested in this paper/area. It performs better than this paper in both attack performance and computational efficiency. The innovation points of these two papers do not overlap, so the audience can try to combine the methods covered in these two papers if a better attack performance is demanded.
