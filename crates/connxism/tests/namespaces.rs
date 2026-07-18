//! Phase 10: namespaces and databases above collections. A `Catalog` routes
//! (namespace, database) to its own estate, so cross-namespace isolation is by
//! construction — the gate proves a query in one namespace cannot see another's
//! data, that the registry lists what exists, and that drop removes a database.

use connxism::Catalog;
use rro_core::{Embedding, Recall, VectorRecord};

fn rec(id: &str) -> VectorRecord {
    VectorRecord::new(id, Embedding(vec![0.1, 0.2, 0.3, 0.4]), format!("doc {id}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn namespaces_are_isolated_and_the_registry_lists_them() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path());

    // Two tenants write into the same-named database under different namespaces.
    let acme = catalog.database("acme", "main").unwrap();
    acme.recall().upsert(vec![rec("acme_doc")]).await.unwrap();

    let globex = catalog.database("globex", "main").unwrap();
    globex
        .recall()
        .upsert(vec![rec("globex_doc")])
        .await
        .unwrap();

    // Isolation: each sees only its own document — no cross-namespace leak.
    assert_eq!(acme.recall().len().await.unwrap(), 1);
    assert_eq!(globex.recall().len().await.unwrap(), 1);
    let q = Embedding(vec![0.1, 0.2, 0.3, 0.4]);
    let acme_hits: Vec<String> = acme
        .recall()
        .search(&q, 10)
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();
    assert_eq!(
        acme_hits,
        vec!["acme_doc"],
        "acme must not see globex's data"
    );
    let globex_hits: Vec<String> = globex
        .recall()
        .search(&q, 10)
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();
    assert_eq!(globex_hits, vec!["globex_doc"]);

    // The registry lists the hierarchy.
    assert_eq!(catalog.namespaces().unwrap(), vec!["acme", "globex"]);
    assert_eq!(catalog.databases("acme").unwrap(), vec!["main"]);

    // Same (ns, db) returns the same cached handle.
    let acme_again = catalog.database("acme", "main").unwrap();
    assert_eq!(acme_again.recall().len().await.unwrap(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn drop_database_removes_it_and_names_are_traversal_safe() {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog::open(dir.path());

    let db = catalog.database("acme", "scratch").unwrap();
    db.recall().upsert(vec![rec("x")]).await.unwrap();
    drop(db); // release the handle so the store can close

    catalog.drop_database("acme", "scratch").unwrap();
    assert!(catalog.databases("acme").unwrap().is_empty());

    // A fresh open of the same path starts empty (the store was removed).
    let db = catalog.database("acme", "scratch").unwrap();
    assert_eq!(db.recall().len().await.unwrap(), 0);

    // Traversal / injection in names is rejected, never touched on disk.
    for bad in ["../escape", "a/b", "", "with space", "a\\b"] {
        assert!(
            catalog.database(bad, "main").is_err(),
            "must reject unsafe namespace name {bad:?}"
        );
    }
}
