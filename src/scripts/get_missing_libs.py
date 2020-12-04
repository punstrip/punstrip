import tqdm

import context
import classes.config
import classes.database
import crf.known_lib_crf


if __name__ == '__main__':
    conf = classes.config.Config()
    db = classes.database.Database(conf)

    missing = set([])

    for path in tqdm.tqdm(db.distinct_binaries()):
        missing_libs = crf.known_lib_crf.missing_libs(db, path)
        if len(missing_libs) > 0:
            print(path)
            print(missing_libs)

            for i in missing_libs:
                if i not in missing:
                    missing.add(i)

    print("Total missing: ")
    print(missing)
    import IPython
    IPython.embed()
