Search.setIndex({"docnames": ["api", "api/components", "api/components/architecture", "api/components/datasplit", "api/components/task", "api/components/trainer", "api/configs", "index", "install", "overview", "tutorial_simple_experiment_dashboard", "tutorial_simple_experiment_python"], "filenames": ["api.rst", "api/components.rst", "api/components/architecture.rst", "api/components/datasplit.rst", "api/components/task.rst", "api/components/trainer.rst", "api/configs.rst", "index.rst", "install.rst", "overview.rst", "tutorial_simple_experiment_dashboard.rst", "tutorial_simple_experiment_python.rst"], "titles": ["API Reference", "Components Reference", "Architecture Reference", "DataSplit Reference", "Task Reference", "Trainer Reference", "Configs", "Welcome to DaCapo\u2019s documentation!", "Installation", "Overview", "Tutorial: A Simple Experiment using the Dashboard", "Tutorial: A Simple Experiment in Python"], "terms": {"compon": [0, 7, 9, 10, 11], "run": [0, 2, 7, 9, 10], "main": [0, 10, 11], "config": [0, 7, 9], "runconfig": [0, 10, 11], "datasplitconfig": 0, "architectureconfig": 0, "taskconfig": 0, "trainerconfig": [0, 10, 11], "class": [1, 2, 3, 4, 5, 6], "dacapo": [1, 2, 3, 4, 5, 6, 8, 10, 11], "experi": [1, 2, 3, 4, 5, 6, 7, 9], "run_config": [1, 11], "static": 1, "get_validation_scor": 1, "method": 1, "avoid": [1, 3], "have": [1, 3, 5, 10, 11], "initi": [1, 5], "model": [1, 3, 5, 9, 10, 11], "optim": [1, 5, 9], "trainer": [1, 6, 9, 11], "etc": [1, 3, 9, 11], "return": [1, 2, 3, 5], "type": [1, 2, 3, 5, 10, 11], "validationscor": 1, "architectur": [1, 6, 9, 11], "datasplit": [1, 6, 9, 11], "dataset": [1, 5, 10, 11], "arrai": [1, 10, 11], "task": [1, 5, 6, 9, 10, 11], "properti": [2, 3], "eval_shape_increas": [2, 6, 11], "coordin": [2, 3, 11], "how": [2, 7, 10, 11], "much": [2, 3, 9, 10, 11], "increas": 2, "input": [2, 3, 5, 10, 11], "shape": 2, "dure": [2, 9, 10, 11], "predict": [2, 3, 6, 10, 11], "abstract": [2, 3, 5], "input_shap": [2, 6, 11], "The": [2, 3, 5, 6, 9, 10, 11], "spatial": [2, 3], "i": [2, 3, 4, 5, 6, 7, 10, 11], "e": [2, 3, 6], "account": 2, "channel": [2, 3, 6], "batch": [2, 9], "dimens": [2, 3], "thi": [2, 3, 4, 5, 6, 9, 10, 11], "num_in_channel": 2, "int": [2, 3], "number": [2, 3, 6, 9], "expect": 2, "num_out_channel": 2, "output": [2, 5, 6, 9, 10, 11], "cnnectomeunet": [2, 6], "architecture_config": [2, 6, 11], "forward": 2, "x": 2, "defin": [2, 3, 5, 9, 10, 11], "comput": [2, 10, 11], "perform": [2, 5, 10, 11], "everi": [2, 10, 11], "call": 2, "should": [2, 3, 11], "overridden": 2, "all": [2, 6, 10, 11], "subclass": 2, "although": 2, "recip": 2, "pass": 2, "need": [2, 3, 5, 9, 10, 11], "within": 2, "function": [2, 9], "one": 2, "modul": [2, 7], "instanc": [2, 6, 9], "afterward": 2, "instead": 2, "sinc": 2, "former": 2, "take": 2, "care": 2, "regist": 2, "hook": 2, "while": 2, "latter": 2, "silent": 2, "ignor": 2, "them": [2, 3, 9, 10, 11], "ar": [3, 5, 9, 10, 11], "collect": 3, "multipl": 3, "each": [3, 6, 9, 10, 11], "assign": 3, "specif": [3, 5, 9, 10, 11], "role": 3, "train": [3, 5, 6, 9, 10, 11], "data": [3, 5, 7, 9], "valid": [3, 6, 9, 10, 11], "test": [3, 4], "trainvalidatedatasplit": [3, 6], "datasplit_config": [3, 6, 11], "configur": [3, 6, 9, 10, 11], "trainvalidatedatasplitconfig": [3, 6], "region": 3, "contain": 3, "necessari": [3, 10, 11], "provid": [3, 9, 11], "can": [3, 5, 6, 9, 10, 11], "includ": [3, 6, 9, 10, 11], "raw": [3, 10, 11], "ground_truth": 3, "mask": 3, "could": [3, 6], "just": [3, 4, 10, 11], "case": [3, 10, 11], "self": 3, "supervis": 3, "abc": 3, "implement": 3, "rawgtdataset": 3, "dataset_config": 3, "rawgtdatasetconfig": 3, "interfac": [3, 11], "contigu": 3, "ground": [3, 10, 11], "truth": [3, 10, 11], "ani": 3, "other": [3, 9, 10, 11], "direct": 3, "some": [3, 5, 9, 10, 11], "storag": [3, 7], "zarr": [3, 10, 11], "n5": 3, "tiff": 3, "stack": 3, "wrapper": 3, "modifi": [3, 10, 11], "anoth": 3, "might": [3, 11], "oper": 3, "normal": [3, 9], "intens": 3, "binar": 3, "label": 3, "gener": [3, 4, 6], "upsampl": [3, 6], "downsampl": [3, 9], "around": 3, "allow": [3, 9, 10, 11], "u": [3, 10, 11], "lazili": 3, "fetch": [3, 10, 11], "transform": [3, 5, 6, 9, 10, 11], "we": [3, 9, 10, 11], "consist": 3, "differ": 3, "context": 3, "attr": 3, "dict": 3, "str": 3, "dictionari": 3, "metadata": 3, "attribut": 3, "store": [3, 9, 10, 11], "ax": 3, "list": 3, "string": 3, "charact": 3, "thei": 3, "index": [3, 7, 10, 11], "permit": 3, "zyx": 3, "c": [3, 6], "": [3, 5], "sampl": 3, "ndarrai": 3, "get": [3, 6, 10, 11], "numpi": 3, "like": 3, "readabl": 3, "writabl": 3, "view": [3, 10, 11], "dim": 3, "dtype": 3, "num_channel": 3, "option": [3, 8, 10, 11], "none": [3, 5, 6], "doesn": [3, 10, 11], "t": [3, 5, 10, 11], "exist": 3, "roi": 3, "total": 3, "world": 3, "unit": 3, "voxel_s": 3, "size": [3, 5, 9], "voxel": [3, 4, 6], "physic": 3, "bool": [3, 5], "write": 3, "zarrarrai": 3, "array_config": 3, "zarrarrayconfig": 3, "classmethod": 3, "create_from_array_identifi": 3, "array_identifi": 3, "write_s": 3, "name": [3, 6, 9, 11], "creat": [3, 5, 7, 10], "new": 3, "given": [3, 5], "an": [3, 6, 9, 10, 11], "identifi": 3, "It": 3, "assum": 3, "point": [3, 9], "doe": [3, 7], "yet": 3, "binarizearrai": 3, "uint": 3, "annot": 3, "becaus": 3, "often": [3, 9], "want": [3, 9, 10, 11], "combin": [3, 9, 11], "set": [3, 5, 10, 11], "wrap": 3, "someth": 3, "group": 3, "mito": 3, "3": [3, 10], "4": [3, 9, 10, 11], "5": [3, 6, 10, 11], "where": [3, 5, 6, 9, 10, 11], "correspond": [3, 10, 11], "mito_membran": 3, "mito_ribo": 3, "everyth": [3, 10, 11], "els": 3, "part": 3, "mitochondria": 3, "simpli": [3, 6], "singl": [3, 9, 10, 11], "binari": 3, "th": 3, "us": [3, 5, 6, 9, 11], "per": [3, 10, 11], "mai": [3, 5, 9, 11], "overlap": 3, "For": [3, 8, 10, 11], "exampl": [3, 10, 11], "you": [3, 6, 9, 10, 11], "had": 3, "membran": 3, "8": [3, 11], "1": [3, 6, 10, 11], "er_membran": 3, "plasma_membran": 3, "now": [3, 11], "classif": 3, "which": 3, "binarizearrayconfig": 3, "resampledarrai": 3, "resampledarrayconfig": 3, "intensitiesarrai": 3, "rang": [3, 10, 11], "0": [3, 6, 11], "convert": 3, "float32": 3, "your": [3, 9, 10, 11], "uint8": 3, "similar": 3, "float": 3, "intensitiesarrayconfig": 3, "missingannotationsmask": 3, "complementari": 3, "individu": 3, "find": [3, 9, 10, 11], "crop": 3, "present": 3, "In": [3, 10, 11], "volum": [3, 9, 10, 11], "see": [3, 11], "packag": 3, "fibsem_tool": 3, "appropri": 3, "format": [3, 9], "indic": 3, "presenc": 3, "http": [3, 6, 8, 10, 11], "github": [3, 6, 8, 11], "com": [3, 6, 8, 11], "janelia": 3, "cosem": 3, "fibsem": 3, "tool": 3, "missintannotationsmaskconfig": 3, "onesarrai": 3, "source_arrai": 3, "ones": 3, "same": [3, 10, 11], "onesarrayconfig": 3, "concatarrai": 3, "concaten": 3, "along": 3, "concatarrayconfig": 3, "logicalorarrai": 3, "logicalorarrayconfig": 3, "croparrai": 3, "larger": 3, "smaller": 3, "croparrayconfig": 3, "onehottask": [4, 6], "task_config": [4, 6, 11], "affinitiestask": [4, 6], "affin": [4, 6, 9], "distancetask": [4, 6], "dummi": 4, "build_batch_provid": 5, "snapshot_contain": 5, "pipelin": 5, "requir": [5, 9, 10, 11], "know": [5, 10, 11], "pull": 5, "from": [5, 9, 10, 11], "inform": 5, "gt": 5, "target": [5, 10, 11], "snapshot": [5, 10, 11], "save": [5, 10, 11], "can_train": 5, "create_optim": 5, "torch": [5, 11], "iter": [5, 9, 10, 11], "num_iter": [5, 6, 11], "devic": 5, "trainingiterationstat": 5, "gunpowdertrain": [5, 6], "trainer_config": [5, 6, 11], "repetit": [6, 11], "validation_interv": [6, 11], "1000": [6, 11], "start_config": 6, "train_config": 6, "validate_config": 6, "standard": 6, "datasplit_typ": 6, "alia": 6, "cnnectomeunetconfig": [6, 11], "fmaps_out": [6, 11], "fmaps_in": [6, 11], "num_fmap": [6, 11], "fmap_inc_factor": [6, 11], "downsample_factor": [6, 11], "kernel_size_down": 6, "kernel_size_up": 6, "upsample_factor": 6, "constant_upsampl": [6, 11], "true": [6, 11], "pad": 6, "base": [6, 11], "saalfeldlab": 6, "cnnectom": 6, "blob": 6, "master": 6, "network": 6, "unet_class": 6, "py": [6, 11], "support": [6, 10, 11], "super": 6, "resolut": 6, "via": 6, "factor": 6, "architecture_typ": 6, "onehottaskconfig": 6, "One": 6, "hot": 6, "probabl": 6, "vector": 6, "length": 6, "ha": [6, 9, 10, 11], "posit": 6, "valu": 6, "l1": 6, "norm": 6, "equal": 6, "post": 6, "process": 6, "extrem": 6, "easi": [6, 9], "argmax": 6, "over": 6, "task_typ": 6, "affinitiestaskconfig": [6, 11], "neighborhood": [6, 11], "lsd": 6, "evalu": [6, 9, 10, 11], "segment": [6, 9, 10, 11], "distancetaskconfig": 6, "clip_dist": 6, "tol_dist": 6, "scale_factor": 6, "mask_dist": 6, "fals": [6, 11], "distanc": [6, 9], "sign": 6, "wai": [6, 10, 11], "advantag": 6, "regular": 6, "denser": 6, "signal": 6, "misclassifi": 6, "pixel": 6, "merg": 6, "2": [6, 10, 11], "otherwis": 6, "veri": [6, 10, 11], "distinct": 6, "object": 6, "cannot": 6, "happen": 6, "gunpowdertrainerconfig": [6, 11], "batch_siz": [6, 11], "learning_r": [6, 11], "num_data_fetch": [6, 11], "augment": [6, 9, 11], "noth": 6, "snapshot_interv": [6, 10, 11], "min_mask": [6, 11], "15": [6, 11], "trainer_typ": 6, "overview": 7, "what": 7, "work": [7, 11], "instal": 7, "api": 7, "refer": 7, "tutori": 7, "A": 7, "simpl": [7, 9], "python": [7, 9], "start": [7, 9, 10], "search": 7, "page": 7, "latest": 8, "version": 8, "pip": 8, "git": 8, "funkelab": [8, 11], "web": [8, 11], "gui": [8, 10, 11], "dashboard": [8, 11], "framework": 9, "execut": [9, 10, 11], "establish": 9, "machin": 9, "learn": [9, 10, 11], "techniqu": 9, "arbitrarili": 9, "larg": [9, 10, 11], "multi": 9, "dimension": 9, "imag": [9, 10, 11], "major": 9, "These": [9, 10, 11], "whether": 9, "scratch": 9, "continu": 9, "off": 9, "previous": 9, "stop": 9, "criterion": 9, "separ": 9, "job": [9, 11], "nice": [9, 10, 11], "structur": 9, "here": [9, 10, 11], "respons": 9, "do": [9, 10], "biomed": 9, "translat": 9, "util": 9, "unet": 9, "even": 9, "after": 9, "choos": [9, 10, 11], "still": 9, "addit": 9, "paramet": [9, 10, 11], "mani": 9, "convolut": 9, "layer": 9, "If": [9, 10, 11], "so": [9, 11], "foreground": 9, "background": 9, "commonli": 9, "loss": [9, 10, 11], "metric": [9, 10, 11], "also": [9, 10, 11], "non": 9, "linear": 9, "loop": 9, "three": 9, "togeth": 9, "sort": 9, "appli": 9, "rate": 9, "give": 9, "uniqu": 9, "mongodb": [9, 10, 11], "filesystem": 9, "retriev": [9, 10, 11], "easili": [9, 10, 11], "multitud": 9, "demonstr": 9, "assembl": 9, "goe": [10, 11], "through": [10, 11], "step": [10, 11], "As": [10, 11], "neuron": [10, 11], "cremi": [10, 11], "org": [10, 11], "3d": [10, 11], "net": [10, 11], "first": [10, 11], "follow": [10, 11], "guid": [10, 11], "make": [10, 11], "sure": 10, "next": [10, 11], "mode": [10, 11], "disk": [10, 11], "particularli": [10, 11], "weight": [10, 11], "sens": [10, 11], "cloud": [10, 11], "dens": [10, 11], "encourag": [10, 11], "score": [10, 11], "quickli": [10, 11], "comparison": [10, 11], "note": [10, 11], "up": [10, 11], "access": [10, 11], "coupl": [10, 11], "stat": [10, 11], "statist": [10, 11], "long": [10, 11], "took": [10, 11], "avail": [10, 11], "ref": [10, 11], "sec_api_run": [10, 11], "held": [10, 11], "out": [10, 11], "n": [10, 11], "interv": [10, 11], "sec_api_runconfig": [10, 11], "qualit": [10, 11], "inspect": [10, 11], "result": [10, 11], "best": [10, 11], "accord": [10, 11], "choic": [10, 11], "checkpoint": [10, 11], "copi": [10, 11], "variou": [10, 11], "let": [10, 11], "sec_api_trainerconfig": [10, 11], "extra": [10, 11], "help": [10, 11], "debug": [10, 11], "gradient": [10, 11], "6": 10, "To": [10, 11], "our": [10, 11], "reproduc": [10, 11], "file": [10, 11], "peopl": [10, 11], "exact": [10, 11], "chang": [10, 11], "compar": [10, 11], "yaml": [10, 11], "templat": [10, 11], "mongodbhost": [10, 11], "dbuser": [10, 11], "dbpass": [10, 11], "dburl": [10, 11], "dbport": [10, 11], "mongodbnam": [10, 11], "runs_base_dir": [10, 11], "path": [10, 11], "my": [10, 11], "host": [10, 11], "databas": [10, 11], "replac": [10, 11], "command": [10, 11], "line": [10, 11], "done": 11, "There": 11, "handl": 11, "basic": 11, "import": 11, "log": 11, "basicconfig": 11, "level": 11, "info": 11, "todo": 11, "small_unet": 11, "216": 11, "72": 11, "32": 11, "affinitiespredict": 11, "gp_augment": 11, "simpleaugmentconfig": 11, "elasticaugmentconfig": 11, "intensityaugmentconfig": 11, "gunpowd": 11, "0001": 11, "control_point_spac": 11, "100": 11, "control_point_displacement_sigma": 11, "10": 11, "rotation_interv": 11, "math": 11, "pi": 11, "subsampl": 11, "uniform_3d_rot": 11, "scale": 11, "25": 11, "75": 11, "shift": 11, "35": 11, "clip": 11, "20": 11, "10000": 11, "min_label": 11, "funlib": 11, "geometri": 11, "tutorial_run": 11, "100000": 11, "summari": 11, "print": 11, "train_run": 11, "haven": 11, "difficult": 11, "without": 11, "past": 11, "ve": 11, "written": 11, "far": 11, "create_stor": 11, "create_config_stor": 11, "config_stor": 11, "store_datasplit_config": 11, "store_architecture_config": 11, "store_task_config": 11, "store_trainer_config": 11, "store_run_config": 11, "onc": 11, "cluster": 11, "r": 11, "conveni": 11, "node": 11, "specifi": 11, "final": 11, "full": 11, "script": 11}, "objects": {"": [[11, 0, 0, "-", "dacapo"]], "dacapo.experiments": [[1, 1, 1, "", "Run"], [6, 1, 1, "", "RunConfig"]], "dacapo.experiments.Run": [[1, 2, 1, "", "get_validation_scores"]], "dacapo.experiments.architectures": [[2, 1, 1, "", "Architecture"], [2, 1, 1, "", "CNNectomeUNet"], [6, 1, 1, "", "CNNectomeUNetConfig"]], "dacapo.experiments.architectures.Architecture": [[2, 3, 1, "", "eval_shape_increase"], [2, 3, 1, "", "input_shape"], [2, 3, 1, "", "num_in_channels"], [2, 3, 1, "", "num_out_channels"]], "dacapo.experiments.architectures.CNNectomeUNet": [[2, 3, 1, "", "eval_shape_increase"], [2, 2, 1, "", "forward"], [2, 3, 1, "", "input_shape"], [2, 3, 1, "", "num_in_channels"], [2, 3, 1, "", "num_out_channels"]], "dacapo.experiments.architectures.CNNectomeUNetConfig": [[6, 4, 1, "", "architecture_type"]], "dacapo.experiments.datasplits": [[3, 1, 1, "", "DataSplit"], [3, 1, 1, "", "TrainValidateDataSplit"], [6, 1, 1, "", "TrainValidateDataSplitConfig"]], "dacapo.experiments.datasplits.TrainValidateDataSplitConfig": [[6, 4, 1, "", "datasplit_type"]], "dacapo.experiments.datasplits.datasets": [[3, 1, 1, "", "Dataset"], [3, 1, 1, "", "RawGTDataset"]], "dacapo.experiments.datasplits.datasets.arrays": [[3, 1, 1, "", "Array"], [3, 1, 1, "", "BinarizeArray"], [3, 1, 1, "", "ConcatArray"], [3, 1, 1, "", "CropArray"], [3, 1, 1, "", "IntensitiesArray"], [3, 1, 1, "", "LogicalOrArray"], [3, 1, 1, "", "MissingAnnotationsMask"], [3, 1, 1, "", "OnesArray"], [3, 1, 1, "", "ResampledArray"], [3, 1, 1, "", "ZarrArray"]], "dacapo.experiments.datasplits.datasets.arrays.Array": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.BinarizeArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.ConcatArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.CropArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.IntensitiesArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.LogicalOrArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.MissingAnnotationsMask": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.OnesArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.ResampledArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.datasplits.datasets.arrays.ZarrArray": [[3, 3, 1, "", "attrs"], [3, 3, 1, "", "axes"], [3, 2, 1, "", "create_from_array_identifier"], [3, 3, 1, "", "data"], [3, 3, 1, "", "dims"], [3, 3, 1, "", "dtype"], [3, 3, 1, "", "num_channels"], [3, 3, 1, "", "roi"], [3, 3, 1, "", "voxel_size"], [3, 3, 1, "", "writable"]], "dacapo.experiments.tasks": [[4, 1, 1, "", "AffinitiesTask"], [6, 1, 1, "", "AffinitiesTaskConfig"], [4, 1, 1, "", "DistanceTask"], [6, 1, 1, "", "DistanceTaskConfig"], [4, 1, 1, "", "OneHotTask"], [6, 1, 1, "", "OneHotTaskConfig"], [4, 1, 1, "", "Task"]], "dacapo.experiments.tasks.AffinitiesTaskConfig": [[6, 4, 1, "", "task_type"]], "dacapo.experiments.tasks.DistanceTaskConfig": [[6, 4, 1, "", "task_type"]], "dacapo.experiments.tasks.OneHotTaskConfig": [[6, 4, 1, "", "task_type"]], "dacapo.experiments.trainers": [[5, 1, 1, "", "GunpowderTrainer"], [6, 1, 1, "", "GunpowderTrainerConfig"], [5, 1, 1, "", "Trainer"]], "dacapo.experiments.trainers.GunpowderTrainer": [[5, 2, 1, "", "build_batch_provider"], [5, 2, 1, "", "can_train"], [5, 2, 1, "", "create_optimizer"], [5, 2, 1, "", "iterate"]], "dacapo.experiments.trainers.GunpowderTrainerConfig": [[6, 4, 1, "", "trainer_type"]], "dacapo.experiments.trainers.Trainer": [[5, 2, 1, "", "build_batch_provider"], [5, 2, 1, "", "can_train"], [5, 2, 1, "", "create_optimizer"], [5, 2, 1, "", "iterate"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:property", "4": "py:attribute"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "property", "Python property"], "4": ["py", "attribute", "Python attribute"]}, "titleterms": {"api": 0, "refer": [0, 1, 2, 3, 4, 5], "compon": 1, "run": [1, 11], "main": 1, "architectur": 2, "datasplit": 3, "dataset": 3, "arrai": 3, "task": 4, "trainer": 5, "config": [6, 10, 11], "runconfig": 6, "datasplitconfig": 6, "architectureconfig": 6, "taskconfig": 6, "trainerconfig": 6, "welcom": 7, "dacapo": [7, 9], "": 7, "document": 7, "content": 7, "indic": 7, "tabl": 7, "instal": [8, 10, 11], "overview": 9, "what": 9, "i": 9, "how": 9, "doe": 9, "work": 9, "tutori": [10, 11], "A": [10, 11], "simpl": [10, 11], "experi": [10, 11], "us": 10, "dashboard": 10, "data": [10, 11], "storag": [10, 11], "python": 11, "creat": 11, "start": 11}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 56}})