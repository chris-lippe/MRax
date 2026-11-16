import { useEffect, useRef } from 'react';
import type { Options } from './api';

export function usePipelineOptions(
  enabled: boolean,
  dirty: boolean,
  options: Options,
  stage?: (options: Options, problem?: string) => void,
  problem?: string,
) {
  const callback = useRef(stage);
  callback.current = stage;
  const value = JSON.stringify(options);
  useEffect(() => {
    if (enabled && dirty) callback.current?.(JSON.parse(value) as Options, problem);
  }, [enabled, dirty, value, problem]);
}
