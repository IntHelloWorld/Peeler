/**
   * Check if resources from a group were changed. If a change is detected, the changeListener will be invoked.
   *
   * @param cacheKey
   *          the cache key which was requested. The key contains the groupName which has to be checked for changes.
   */
  public void check(final CacheKey cacheKey, final Callback callback) {
    notNull(cacheKey);
    LOG.debug("started");
    final StopWatch watch = new StopWatch();
    watch.start("detect changes");
    try {
        final Group group = new WroModelInspector(modelFactory.create()).getGroupByName(cacheKey.getGroupName());
        if (isGroupChanged(group.collectResourcesOfType(cacheKey.getType()), callback)) {
            callback.onGroupChanged(cacheKey);
            cacheStrategy.put(cacheKey, null);
        }
        resourceChangeDetector.reset();
    } catch (final Exception e) {
        onException(e);
    } finally {
        watch.stop();
        LOG.debug("resource watcher info: {}", watch.prettyPrint());
    }
}