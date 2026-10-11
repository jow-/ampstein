// Phase 0 spike 0.1: prove torch.so loads and calls into libtorch.
//
// Run:  ucode -L <build-dir> examples/version.uc
import { version } from 'ampstein.torch';

printf('%s\n', version());