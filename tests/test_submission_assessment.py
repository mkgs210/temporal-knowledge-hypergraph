import numpy as np
import pytest


def test_independent_coherence_uses_type_matched_null_and_complete_partition():
    from tkh.submission_assessment import independent_coherence
    nodes=[{'id':str(i),'type':'method'} for i in range(4)]
    groups=[{'id':'a','members':['0','1']},{'id':'b','members':['2','3']}]
    vectors=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    result=independent_coherence(groups,nodes,vectors)
    assert result['observed']==pytest.approx(1.)
    assert result['within_group_pairs']==2
    assert len(result['null_values'])==20
    assert result['excess']>0
    with pytest.raises(ValueError,match='partition'):
        independent_coherence(groups[:1],nodes,vectors)
