from datetime import datetime

from network_creating import (
    adj_from_file,
    bbcode_adjacency,
    kagome_adjacency,
    lattice_adjacency,
)
from testing import test_plots

#timestamp so that files dont get overwritten
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# if you want to oberwrite previous files, just set timestamp to an empty string
# timestamp = ""

#figure 8 and S14 
def plots(printing=True, use_precomputed=True):
    # square
    test_plots(
        adj=lattice_adjacency(18),
        filename=f"{timestamp}_18by18_square_d0",
        l_0_values=[0, 2, 4],
        quantities=["marginals", "U"],
        use_precomputed=use_precomputed,
        normalized_tn=True,
        printing=printing,
    )
    # power grid
    test_plots(
        adj=adj_from_file("sources/power.txt"),
        filename=f"{timestamp}_power",
        l_0_values=[0, 1, 2],
        quantities=["marginals", "S", "Z"],
        use_precomputed=use_precomputed,
        normalized_tn=True,
        printing=printing,
    )
    test_plots(
        adj=bbcode_adjacency(12, offset=1),
        filename=f"{timestamp}_12by12_square_d1",
        l_0_values=[0, 1, 2],
        quantities=["marginals", "U"],
        use_precomputed=use_precomputed,
        normalized_tn=False,
        printing=printing,
    )
    test_plots(
        adj=bbcode_adjacency(12, offset=2),
        filename=f"{timestamp}_12by12_square_d2",
        l_0_values=[0, 2, 3],
        quantities=["marginals"],
        use_precomputed=use_precomputed,
        normalized_tn=False,
        printing=printing,
    )
    
    #kagome
    test_plots(
        adj=kagome_adjacency(16),
        filename=f"{timestamp}_16by16_kagome",
        l_0_values=[0, 1, 4, 5, 6],
        quantities=["marginals", "Z"],
        use_precomputed=use_precomputed,
        normalized_tn=False,
        printing=printing,
    )
    
#the source of the runtime plots, here we compute all values
#figure 7
def runtime():
    test_plots(
        adj=adj_from_file("sources/power.txt"),
        filename=f"{timestamp}_power_runtime",
        l_0_values=[0, 1, 2, 3],
        use_precomputed=True,
        normalized_tn=True,
        printing=False,
        only_runtime=True,
    )
    # square
    test_plots(
        adj=lattice_adjacency(18),
        filename=f"{timestamp}_18by18_square_d0_runtime",
        l_0_values=[0, 2, 4],
        use_precomputed=True,
        normalized_tn=True,
        printing=False,
        only_runtime=True,
    )
    test_plots(
        adj=bbcode_adjacency(12, offset=1),
        filename=f"{timestamp}_12by12_square_d1_runtime",
        l_0_values=[0, 1, 2],
        use_precomputed=True,
        normalized_tn=True,
        printing=False,
        only_runtime=True,
    )
    test_plots(
        adj=bbcode_adjacency(12, offset=2),
        filename=f"{timestamp}_12by12_square_d2_runtime",
        l_0_values=[0, 2, 3],
        use_precomputed=True,
        normalized_tn=True,
        printing=False,
        only_runtime=True,
    )
    
    #kagome
    test_plots(
        adj=kagome_adjacency(16),
        filename=f"{timestamp}_16by16_kagome_runtime",
        l_0_values=[0, 1, 4, 5, 6],
        use_precomputed=True,
        normalized_tn=True,
        printing=False,
        only_runtime=True,
    )
    
if __name__ == "__main__":
    plots()
    #runtime()
    
