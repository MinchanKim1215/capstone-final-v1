"""Count-header repair must preserve inputs and every evaluation outcome."""
from pathlib import Path
import unittest
import uuid
from . import ARTIFACTS
from . import metadata
from .contracts import read,write,sha


class CountHeaderRepair(unittest.TestCase):
    def setUp(self):
        self.original_root=metadata.ARTIFACTS
        self.root=ARTIFACTS/'preflight'/('metadata_fixture_'+uuid.uuid4().hex)
        self.root.resolve().relative_to(ARTIFACTS.resolve());self.root.mkdir(parents=True)
        metadata.ARTIFACTS=self.root
        old_counts={'train':1,'confirmation':0};rows=[dict(role='train',key='a'),dict(role='confirmation',key='b')]
        write(self.root/'dataset.json',dict(counts=old_counts,rows=rows[:1]))
        self.dataset=self.root/'dataset_with_confirmation.json';write(self.dataset,dict(counts=old_counts,rows=rows))
        self.original_data=read(self.dataset);self.original_summaries={};outcomes=[]
        for tag in ('one','two'):
            path=self.root/(tag+'.json');value=dict(model_tag=tag,dataset_sha256=sha(self.dataset),
                dataset_path=str(self.dataset),metrics=dict(ap=.4321,first_alerts=73,brier=.012345),cases=[dict(key='b',truth_sha='fixed')])
            write(path,value);self.original_summaries[str(path)]=value;outcomes.append(dict(tag=tag,raw_summary=str(path)))
        self.completed=dict(outcomes=outcomes)

    def tearDown(self):
        metadata.ARTIFACTS=self.original_root

    def test_counts_repaired_without_metric_or_input_changes_and_idempotent(self):
        metadata.correct_confirmation_counts(self.completed)
        corrected=read(self.dataset)
        self.assertEqual(corrected['counts'],{'train':1,'confirmation':1})
        self.assertEqual(corrected['rows'],self.original_data['rows'])
        for path,original in self.original_summaries.items():
            current=read(path)
            self.assertEqual(current['dataset_sha256'],sha(self.dataset))
            for key,value in original.items():
                if key!='dataset_sha256':self.assertEqual(current[key],value)
        hashes={path:sha(path) for path in self.original_summaries}
        metadata.correct_confirmation_counts(self.completed)
        self.assertEqual(hashes,{path:sha(path) for path in hashes})

    def test_unrelated_input_change_is_rejected(self):
        metadata.correct_confirmation_counts(self.completed)
        changed=read(self.dataset);changed['rows'][1]['key']='changed';write(self.dataset,changed)
        with self.assertRaises(ValueError):metadata.correct_confirmation_counts(self.completed)


if __name__=='__main__':unittest.main()
