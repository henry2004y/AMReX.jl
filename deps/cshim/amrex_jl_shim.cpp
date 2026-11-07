#include <jlcxx/jlcxx.hpp>
#include <jlcxx/stl.hpp>

#include <AMReX.H>
#include <AMReX_AmrCore.H>
#include <AMReX_AmrParticles.H>
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Particle.H>
#include <AMReX_ParticleTile.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_RealBox.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

template <class FAB>
class PhysBCFunctNoOpFab {
public:
    void operator()(amrex::FabArray<FAB>& /*mf*/, int /*dcomp*/, int /*ncomp*/,
                    amrex::IntVect const& /*nghost*/, amrex::Real /*time*/,
                    int /*bccomp*/) {}
};

std::vector<std::string> split_varnames(const std::string& joined, int ncomp) {
    std::vector<std::string> res;
    if (!joined.empty()) {
        std::stringstream ss(joined);
        std::string item;
        while (std::getline(ss, item, ',')) {
            res.push_back(item);
        }
    }
    while (static_cast<int>(res.size()) < ncomp) {
        res.push_back("var_" + std::to_string(res.size() + 1));
    }
    return res;
}

amrex::Interpolater* get_interpolater(int interp_type) {
    switch (interp_type) {
        case 0: return &amrex::cell_cons_interp;
        case 1: return &amrex::node_bilinear_interp;
        case 2: return &amrex::pc_interp;
        case 3: return &amrex::quadratic_interp;
        default: return &amrex::cell_cons_interp;
    }
}

} // anonymous namespace

// ============================================================================
// Wrapper classes
// ============================================================================

struct JlBox {
    amrex::Box box;

    JlBox() = default;
    explicit JlBox(const amrex::Box& b) : box(b) {}
    JlBox(int lox, int loy, int loz, int hix, int hiy, int hiz)
        : box(amrex::IntVect(lox, loy, loz), amrex::IntVect(hix, hiy, hiz)) {}
    JlBox(int lox, int loy, int loz, int hix, int hiy, int hiz,
          int tx, int ty, int tz)
        : box(amrex::IntVect(lox, loy, loz), amrex::IntVect(hix, hiy, hiz),
              amrex::IntVect(tx, ty, tz)) {}

    int lo(int dir) const { return box.smallEnd(dir); }
    int hi(int dir) const { return box.bigEnd(dir); }
    int length(int dir) const { return box.length(dir); }
    bool is_nodal(int dir) const { return box.type(dir) == amrex::IndexType::NODE; }
    int64_t num_pts() const { return box.numPts(); }
    bool is_empty() const { return !box.ok(); }
    bool contains(int i, int j, int k) const {
        return box.contains(amrex::IntVect(i, j, k));
    }
    JlBox surrounding_nodes() const {
        return JlBox(amrex::surroundingNodes(box));
    }
    JlBox enclosed_cells() const {
        return JlBox(amrex::enclosedCells(box));
    }
    JlBox grow(int n) const {
        amrex::Box b = box;
        b.grow(n);
        return JlBox(b);
    }
    JlBox intersect(const JlBox& other) const {
        return JlBox(box & other.box);
    }
};

struct JlRealBox {
    amrex::RealBox rbox;

    JlRealBox() = default;
    explicit JlRealBox(const amrex::RealBox& rb) : rbox(rb) {}
    JlRealBox(double xlo, double ylo, double zlo, double xhi, double yhi, double zhi)
        : rbox({xlo, ylo, zlo}, {xhi, yhi, zhi}) {}

    double lo(int dir) const { return rbox.lo(dir); }
    double hi(int dir) const { return rbox.hi(dir); }
    double length(int dir) const { return rbox.length(dir); }
    bool contains(double x, double y, double z) const {
        amrex::Real pt[3] = {x, y, z};
        return rbox.contains(pt);
    }
};

struct JlGeometry {
    amrex::Geometry geom;

    JlGeometry() = default;
    explicit JlGeometry(const amrex::Geometry& g) : geom(g) {}
    JlGeometry(const JlBox& dom, const JlRealBox& rb, int coord,
               bool px, bool py, bool pz) {
        amrex::Array<int, AMREX_SPACEDIM> is_per{int(px), int(py), int(pz)};
        geom = amrex::Geometry(dom.box, rb.rbox, coord, is_per);
    }

    double cell_size(int dir) const { return geom.CellSize(dir); }
    double inv_cell_size(int dir) const { return geom.InvCellSize(dir); }
    double prob_lo(int dir) const { return geom.ProbLo(dir); }
    double prob_hi(int dir) const { return geom.ProbHi(dir); }
    JlBox domain() const { return JlBox(geom.Domain()); }
    JlRealBox prob_domain() const { return JlRealBox(geom.ProbDomain()); }
    bool is_periodic(int dir) const { return geom.isPeriodic(dir); }
};

struct JlBoxArray {
    amrex::BoxArray ba;

    JlBoxArray() = default;
    explicit JlBoxArray(const amrex::BoxArray& b) : ba(b) {}
    explicit JlBoxArray(const JlBox& bx) : ba(bx.box) {}

    void max_size(int mx, int my, int mz) {
        ba.maxSize(amrex::IntVect(mx, my, mz));
    }
    JlBoxArray convert_to_nodal(int nx, int ny, int nz) const {
        return JlBoxArray(amrex::convert(ba, amrex::IntVect(nx, ny, nz)));
    }
    int64_t size() const { return ba.size(); }
    bool empty() const { return ba.empty(); }
    JlBox get_box(int64_t i) const { return JlBox(ba[i]); }
    bool contains_cell(int i, int j, int k) const {
        return ba.contains(amrex::IntVect(i, j, k));
    }
};

struct JlDistributionMapping {
    amrex::DistributionMapping dm;

    JlDistributionMapping() = default;
    explicit JlDistributionMapping(const amrex::DistributionMapping& d) : dm(d) {}
    explicit JlDistributionMapping(const JlBoxArray& ba) : dm(ba.ba) {}

    int64_t size() const { return dm.size(); }
    int get_rank(int64_t i) const { return dm[i]; }
};

struct JlMultiFab {
    std::unique_ptr<amrex::MultiFab> mf;

    JlMultiFab() : mf(std::make_unique<amrex::MultiFab>()) {}
    JlMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
               int ncomp, int nghost)
        : mf(std::make_unique<amrex::MultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(nghost))) {}
    JlMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
               int ncomp, int ngx, int ngy, int ngz)
        : mf(std::make_unique<amrex::MultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(ngx, ngy, ngz))) {}

    ~JlMultiFab() {
        if (!amrex::Initialized()) {
            mf.release();
        }
    }

    int n_comp() const { return mf->nComp(); }
    int n_grow(int dir) const { return mf->nGrow(dir); }
    int64_t size() const { return mf->size(); }
    int64_t local_size() const { return mf->local_size(); }
    JlBoxArray box_array() const { return JlBoxArray(mf->boxArray()); }
    JlDistributionMapping dmap() const { return JlDistributionMapping(mf->DistributionMap()); }

    void set_val(double val) { mf->setVal(val); }
    void set_val_comp(double val, int icomp, int ncomp, int nghost) {
        mf->setVal(val, icomp, ncomp, nghost);
    }
    void plus_val(double val, int icomp, int ncomp, int nghost) {
        mf->plus(val, icomp, ncomp, nghost);
    }
    void mult_val(double val, int icomp, int ncomp, int nghost) {
        mf->mult(val, icomp, ncomp, nghost);
    }
    void add_from(const JlMultiFab& src, int scomp, int dcomp, int ncomp, int nghost) {
        amrex::MultiFab::Add(*mf, *src.mf, scomp, dcomp, ncomp, nghost);
    }
    void copy_from(const JlMultiFab& src, int scomp, int dcomp, int ncomp,
                   int nghost) {
        amrex::MultiFab::Copy(*mf, *src.mf, scomp, dcomp, ncomp,
                              amrex::IntVect(nghost));
    }
    void parallel_copy_from(const JlMultiFab& src, int scomp, int dcomp, int ncomp,
                            int snghost, int dnghost, const JlGeometry& geom) {
        mf->ParallelCopy(*src.mf, scomp, dcomp, ncomp,
                         amrex::IntVect(snghost), amrex::IntVect(dnghost),
                         geom.geom.periodicity());
    }
    void fill_boundary(const JlGeometry& geom) {
        mf->FillBoundary(geom.geom.periodicity());
    }
    void fill_boundary_comp(int icomp, int ncomp, const JlGeometry& geom) {
        mf->FillBoundary(icomp, ncomp, geom.geom.periodicity());
    }
    void sum_boundary(const JlGeometry& geom) {
        mf->SumBoundary(geom.geom.periodicity());
    }
    void sum_boundary_comp(int icomp, int ncomp, int snghost, int dnghost,
                           const JlGeometry& geom) {
        mf->SumBoundary(icomp, ncomp, amrex::IntVect(snghost),
                        amrex::IntVect(dnghost), geom.geom.periodicity());
    }
    double min_val(int comp, int nghost) const { return mf->min(comp, nghost); }
    double max_val(int comp, int nghost) const { return mf->max(comp, nghost); }
    double norm0(int comp, int nghost) const { return mf->norm0(comp, nghost); }
    double norm1(int comp, const JlGeometry& geom) const {
        return mf->norm1(comp, geom.geom.periodicity());
    }
    double norm2(int comp, const JlGeometry& geom) const {
        return mf->norm2(comp, geom.geom.periodicity());
    }
    double sum_val(int comp, bool local) const { return mf->sum(comp, local); }
};

struct JliMultiFab {
    std::unique_ptr<amrex::iMultiFab> imf;

    JliMultiFab() : imf(std::make_unique<amrex::iMultiFab>()) {}
    JliMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
                int ncomp, int nghost)
        : imf(std::make_unique<amrex::iMultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(nghost))) {}
    JliMultiFab(const JlBoxArray& ba, const JlDistributionMapping& dm,
                int ncomp, int ngx, int ngy, int ngz)
        : imf(std::make_unique<amrex::iMultiFab>(
              ba.ba, dm.dm, ncomp, amrex::IntVect(ngx, ngy, ngz))) {}

    ~JliMultiFab() {
        if (!amrex::Initialized()) {
            imf.release();
        }
    }

    int n_comp() const { return imf->nComp(); }
    int n_grow(int dir) const { return imf->nGrow(dir); }
    int64_t size() const { return imf->size(); }
    int64_t local_size() const { return imf->local_size(); }
    JlBoxArray box_array() const { return JlBoxArray(imf->boxArray()); }
    JlDistributionMapping dmap() const { return JlDistributionMapping(imf->DistributionMap()); }

    void set_val(int val) { imf->setVal(val); }
    void copy_from(const JliMultiFab& src, int scomp, int dcomp, int ncomp,
                   int nghost) {
        amrex::iMultiFab::Copy(*imf, *src.imf, scomp, dcomp, ncomp,
                               amrex::IntVect(nghost));
    }
    void fill_boundary(const JlGeometry& geom) {
        imf->FillBoundary(geom.geom.periodicity());
    }
    void sum_boundary(const JlGeometry& geom) {
        imf->SumBoundary(geom.geom.periodicity());
    }
    int min_val(int comp, int nghost) const { return imf->min(comp, nghost); }
    int max_val(int comp, int nghost) const { return imf->max(comp, nghost); }
};

struct JlMFIter {
    std::unique_ptr<amrex::MFIter> mfi;

    JlMFIter(const JlMultiFab& mf, bool do_tiling)
        : mfi(std::make_unique<amrex::MFIter>(*mf.mf, do_tiling)) {}
    JlMFIter(const JliMultiFab& imf, bool do_tiling)
        : mfi(std::make_unique<amrex::MFIter>(*imf.imf, do_tiling)) {}

    bool is_valid() const { return mfi && mfi->isValid(); }
    void next() { ++(*mfi); }
    void close() { mfi.reset(); }
    int index() const { return mfi->index(); }
    int local_tile_index() const { return mfi->LocalTileIndex(); }
    JlBox valid_box() const { return JlBox(mfi->validbox()); }
    JlBox tile_box() const { return JlBox(mfi->tilebox()); }
    JlBox fab_box() const { return JlBox(mfi->fabbox()); }

    int fab_lo(const JlMultiFab& mf, int dir) const {
        return (*mf.mf)[*mfi].box().smallEnd(dir);
    }
    int fab_hi(const JlMultiFab& mf, int dir) const {
        return (*mf.mf)[*mfi].box().bigEnd(dir);
    }
    int ifab_lo(const JliMultiFab& imf, int dir) const {
        return (*imf.imf)[*mfi].box().smallEnd(dir);
    }
    int ifab_hi(const JliMultiFab& imf, int dir) const {
        return (*imf.imf)[*mfi].box().bigEnd(dir);
    }

    double* data_ptr(JlMultiFab& mf) const {
        return (*mf.mf)[*mfi].dataPtr();
    }
    int* idata_ptr(JliMultiFab& imf) const {
        return (*imf.imf)[*mfi].dataPtr();
    }
};

// ============================================================================
// AmrCore subclass for multi-level AMR hierarchy management
// ============================================================================

class AmrCoreImpl : public amrex::AmrCore {
private:
    static amrex::Geometry make_base_geom(const JlRealBox& rb, int nx, int ny, int nz,
                                          int coord, bool px, bool py, bool pz) {
        amrex::Box dom(amrex::IntVect(0, 0, 0), amrex::IntVect(nx - 1, ny - 1, nz - 1));
        amrex::Array<int, AMREX_SPACEDIM> is_per{int(px), int(py), int(pz)};
        return amrex::Geometry(dom, rb.rbox, coord, is_per);
    }

    static amrex::AmrInfo make_amr_info(int max_level_in, int nz,
                                        int max_grid_x, int max_grid_y, int max_grid_z,
                                        int block_x, int block_y, int block_z,
                                        double grid_eff) {
        amrex::AmrInfo info;
        info.max_level = max_level_in;
        info.ref_ratio.assign(std::max(1, max_level_in),
                              amrex::IntVect(2, 2, (nz == 1) ? 1 : 2));
        info.blocking_factor.assign(max_level_in + 1,
                                    amrex::IntVect(block_x, block_y, (nz == 1) ? 1 : block_z));
        info.max_grid_size.assign(max_level_in + 1,
                                  amrex::IntVect(max_grid_x, max_grid_y, max_grid_z));
        info.grid_eff = grid_eff;
        return info;
    }

public:
    std::vector<std::vector<amrex::RealBox>> refine_boxes;

    AmrCoreImpl(const JlRealBox& rb, int max_level_in,
                int nx, int ny, int nz, int coord,
                bool px, bool py, bool pz,
                int max_grid_x, int max_grid_y, int max_grid_z,
                int block_x, int block_y, int block_z,
                double grid_eff)
        : amrex::AmrCore(
              make_base_geom(rb, nx, ny, nz, coord, px, py, pz),
              make_amr_info(max_level_in, nz, max_grid_x, max_grid_y, max_grid_z,
                            block_x, block_y, block_z, grid_eff)) {
        refine_boxes.resize(std::max(1, max_level_in));
    }

    ~AmrCoreImpl() override = default;

    void set_box_array_pub(int lev, const amrex::BoxArray& ba) { SetBoxArray(lev, ba); }
    void set_dmap_pub(int lev, const amrex::DistributionMapping& dm) { SetDistributionMap(lev, dm); }
    void set_finest_level_pub(int lev) { SetFinestLevel(lev); }

protected:
    void MakeNewLevelFromScratch(int lev, amrex::Real /*time*/,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override {
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);
    }

    void MakeNewLevelFromCoarse(int lev, amrex::Real /*time*/,
                                const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override {
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);
    }

    void RemakeLevel(int lev, amrex::Real /*time*/,
                     const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm) override {
        SetBoxArray(lev, ba);
        SetDistributionMap(lev, dm);
    }

    void ClearLevel(int lev) override {
        ClearBoxArray(lev);
        ClearDistributionMap(lev);
    }

    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/,
                  int /*ngrow*/) override {
        if (lev >= static_cast<int>(refine_boxes.size())) return;
        const auto& geom_lev = Geom(lev);
        for (amrex::MFIter mfi(tags); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto tag_arr = tags.array(mfi);
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        amrex::Real xyz[3];
                        geom_lev.CellCenter({i, j, k}, xyz);
                        for (int rlev = lev; rlev < static_cast<int>(refine_boxes.size()); ++rlev) {
                            for (const auto& rbox : refine_boxes[rlev]) {
                                if (rbox.contains(xyz)) {
                                    tag_arr(i, j, k) = amrex::TagBox::SET;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

struct JlAmrCore {
    std::unique_ptr<AmrCoreImpl> core;

    JlAmrCore(const JlRealBox& rb, int max_level_in,
              int nx, int ny, int nz, int coord,
              bool px, bool py, bool pz,
              int max_grid_x, int max_grid_y, int max_grid_z,
              int block_x, int block_y, int block_z,
              double grid_eff)
        : core(std::make_unique<AmrCoreImpl>(
              rb, max_level_in, nx, ny, nz, coord, px, py, pz,
              max_grid_x, max_grid_y, max_grid_z, block_x, block_y, block_z, grid_eff)) {}

    ~JlAmrCore() {
        if (!amrex::Initialized()) {
            core.release();
        }
    }

    void add_refine_box(int lev, double xlo, double ylo, double zlo,
                        double xhi, double yhi, double zhi) {
        if (lev >= 0 && lev < static_cast<int>(core->refine_boxes.size())) {
            core->refine_boxes[lev].emplace_back(
                amrex::RealBox({xlo, ylo, zlo}, {xhi, yhi, zhi}));
        }
    }

    void clear_refine_boxes() {
        for (auto& v : core->refine_boxes) {
            v.clear();
        }
    }

    void init_from_scratch(double time) { core->InitFromScratch(time); }
    void regrid_levels(int base_lev, double time) { core->regrid(base_lev, time); }

    int get_finest_level() const { return core->finestLevel(); }
    int get_max_level() const { return core->maxLevel(); }
    JlGeometry get_geom(int lev) const { return JlGeometry(core->Geom(lev)); }
    JlBoxArray get_box_array(int lev) const { return JlBoxArray(core->boxArray(lev)); }
    JlBoxArray get_node_box_array(int lev) const {
        return JlBoxArray(amrex::convert(core->boxArray(lev), amrex::IntVect::TheNodeVector()));
    }
    JlDistributionMapping get_dmap(int lev) const {
        return JlDistributionMapping(core->DistributionMap(lev));
    }
    int get_ref_ratio(int lev, int dir) const {
        return (lev < core->maxLevel()) ? core->refRatio(lev)[dir] : 1;
    }

    void set_box_array(int lev, const JlBoxArray& ba) { core->set_box_array_pub(lev, ba.ba); }
    void set_dmap(int lev, const JlDistributionMapping& dm) {
        core->set_dmap_pub(lev, dm.dm);
    }
    void set_finest_level(int lev) { core->set_finest_level_pub(lev); }
};

// ============================================================================
// Pure-SoA ParticleContainer & ParIter
// ============================================================================

using PureSoAParticleContainer =
    amrex::AmrParticleContainer_impl<amrex::SoAParticle<AMREX_SPACEDIM, 0>,
                                     AMREX_SPACEDIM, 0>;
using PureSoAParIter = amrex::ParIterSoA<AMREX_SPACEDIM, 0>;

struct JlParticleContainer {
    std::unique_ptr<PureSoAParticleContainer> pc;
    int num_runtime_real = 0;
    int num_runtime_int = 0;

    JlParticleContainer(JlAmrCore& amr_core, int n_rt_real, int n_rt_int)
        : pc(std::make_unique<PureSoAParticleContainer>(amr_core.core.get())),
          num_runtime_real(n_rt_real),
          num_runtime_int(n_rt_int) {
        pc->SetSoACompileTimeNames({"x", "y", "z"}, {});
        for (int i = 0; i < n_rt_real; ++i) {
            pc->AddRealComp("r_" + std::to_string(i), true);
        }
        for (int i = 0; i < n_rt_int; ++i) {
            pc->AddIntComp("i_" + std::to_string(i), true);
        }
        pc->reserveData();
        pc->resizeData();
    }

    JlParticleContainer(const JlGeometry& geom, const JlDistributionMapping& dm,
                        const JlBoxArray& ba, int n_rt_real, int n_rt_int)
        : pc(std::make_unique<PureSoAParticleContainer>(
              amrex::Vector<amrex::Geometry>{geom.geom},
              amrex::Vector<amrex::DistributionMapping>{dm.dm},
              amrex::Vector<amrex::BoxArray>{ba.ba},
              amrex::Vector<int>{})),
          num_runtime_real(n_rt_real),
          num_runtime_int(n_rt_int) {
        pc->SetSoACompileTimeNames({"x", "y", "z"}, {});
        for (int i = 0; i < n_rt_real; ++i) {
            pc->AddRealComp("r_" + std::to_string(i), true);
        }
        for (int i = 0; i < n_rt_int; ++i) {
            pc->AddIntComp("i_" + std::to_string(i), true);
        }
        pc->reserveData();
        pc->resizeData();
    }

    ~JlParticleContainer() {
        if (!amrex::Initialized()) {
            pc.release();
        }
    }

    void reserve_and_resize_data() {
        pc->reserveData();
        pc->resizeData();
    }

    // Ensure a tile exists at (lev, grid_id, tile_id) and return its current particle count
    int64_t ensure_tile(int lev, int grid_id, int tile_id) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.numParticles();
    }

    void resize_tile(int lev, int grid_id, int tile_id, int64_t new_size) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        ptile.resize(new_size);
    }

    int64_t tile_num_particles(int lev, int grid_id, int tile_id) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.numParticles();
    }

    uint64_t* tile_idcpu_ptr(int lev, int grid_id, int tile_id) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetIdCPUData().dataPtr();
    }

    double* tile_pos_ptr(int lev, int grid_id, int tile_id, int dir) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetRealData(dir).dataPtr();
    }

    double* tile_real_ptr(int lev, int grid_id, int tile_id, int comp) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + comp).dataPtr();
    }

    int* tile_int_ptr(int lev, int grid_id, int tile_id, int comp) {
        auto& ptile = pc->DefineAndReturnParticleTile(lev, grid_id, tile_id);
        return ptile.GetStructOfArrays().GetIntData(comp).dataPtr();
    }

    void redistribute(int lev_min, int lev_max, int nghost) {
        pc->Redistribute(lev_min, lev_max, nghost);
    }

    int64_t total_number_of_particles(bool only_valid, bool local) const {
        return pc->TotalNumberOfParticles(only_valid, local);
    }

    int64_t number_of_particles_at_level(int lev, bool only_valid, bool local) const {
        return pc->NumberOfParticlesAtLevel(lev, only_valid, local);
    }

    void clear_particles() {
        pc->clearParticles();
    }

    void write_plotfile(const std::string& dir, const std::string& name) const {
        pc->WritePlotFile(dir, name);
    }

    void checkpoint(const std::string& dir, const std::string& name) const {
        pc->Checkpoint(dir, name);
    }
};

struct JlParIter {
    std::unique_ptr<PureSoAParIter> pti;

    JlParIter(JlParticleContainer& pc, int lev)
        : pti(std::make_unique<PureSoAParIter>(*pc.pc, lev)) {}

    bool is_valid() const { return pti && pti->isValid(); }
    void next() { ++(*pti); }
    void close() { pti.reset(); }
    int index() const { return pti->index(); }
    int local_tile_index() const { return pti->LocalTileIndex(); }
    JlBox valid_box() const { return JlBox(pti->validbox()); }
    JlBox tile_box() const { return JlBox(pti->tilebox()); }
    int64_t num_particles() const { return pti->numParticles(); }

    void resize_tile(int64_t new_size) {
        pti->GetParticleTile().resize(new_size);
    }

    uint64_t* idcpu_ptr() {
        return pti->GetStructOfArrays().GetIdCPUData().dataPtr();
    }

    double* pos_ptr(int dir) {
        return pti->GetStructOfArrays().GetRealData(dir).dataPtr();
    }

    double* real_ptr(int comp) {
        return pti->GetStructOfArrays().GetRealData(AMREX_SPACEDIM + comp).dataPtr();
    }

    int* int_ptr(int comp) {
        return pti->GetStructOfArrays().GetIntData(comp).dataPtr();
    }
};

// ============================================================================
// JLCXX Module Registration
// ============================================================================

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    // Lifecycle & ParallelDescriptor
    mod.method("amrex_init", [](bool use_parmparse) {
        if (!amrex::Initialized()) {
            int argc = 0;
            char** argv = nullptr;
            amrex::Initialize(argc, argv, use_parmparse);
        }
        amrex::MFIter::allowMultipleMFIters(true);
    });
    mod.method("amrex_initialized", []() { return amrex::Initialized(); });
    mod.method("amrex_finalize", []() {
        if (amrex::Initialized()) {
            amrex::Finalize();
        }
    });
    mod.method("parallel_myproc", []() { return amrex::ParallelDescriptor::MyProc(); });
    mod.method("parallel_nprocs", []() { return amrex::ParallelDescriptor::NProcs(); });
    mod.method("parallel_ioprocessor", []() { return amrex::ParallelDescriptor::IOProcessor(); });
    mod.method("parallel_barrier", []() { amrex::ParallelDescriptor::Barrier(); });

    mod.method("parallel_reduce_sum_real", [](double val) {
        amrex::ParallelDescriptor::ReduceRealSum(val);
        return val;
    });
    mod.method("parallel_reduce_max_real", [](double val) {
        amrex::ParallelDescriptor::ReduceRealMax(val);
        return val;
    });
    mod.method("parallel_reduce_min_real", [](double val) {
        amrex::ParallelDescriptor::ReduceRealMin(val);
        return val;
    });
    mod.method("parallel_reduce_sum_long", [](int64_t val) {
        amrex::Long lval = val;
        amrex::ParallelDescriptor::ReduceLongSum(lval);
        return static_cast<int64_t>(lval);
    });
    mod.method("parallel_reduce_sum_real_vec!", [](jlcxx::ArrayRef<double> arr) {
        amrex::ParallelDescriptor::ReduceRealSum(arr.data(), arr.size());
    });

    // JlBox
    mod.add_type<JlBox>("JlBox")
        .constructor<>()
        .constructor<int, int, int, int, int, int>()
        .constructor<int, int, int, int, int, int, int, int, int>()
        .method("lo", &JlBox::lo)
        .method("hi", &JlBox::hi)
        .method("length", &JlBox::length)
        .method("is_nodal", &JlBox::is_nodal)
        .method("num_pts", &JlBox::num_pts)
        .method("is_empty", &JlBox::is_empty)
        .method("contains", &JlBox::contains)
        .method("surrounding_nodes", &JlBox::surrounding_nodes)
        .method("enclosed_cells", &JlBox::enclosed_cells)
        .method("grow", &JlBox::grow)
        .method("intersect", &JlBox::intersect);

    // JlRealBox
    mod.add_type<JlRealBox>("JlRealBox")
        .constructor<>()
        .constructor<double, double, double, double, double, double>()
        .method("lo", &JlRealBox::lo)
        .method("hi", &JlRealBox::hi)
        .method("length", &JlRealBox::length)
        .method("contains", &JlRealBox::contains);

    // JlGeometry
    mod.add_type<JlGeometry>("JlGeometry")
        .constructor<>()
        .constructor<const JlBox&, const JlRealBox&, int, bool, bool, bool>()
        .method("cell_size", &JlGeometry::cell_size)
        .method("inv_cell_size", &JlGeometry::inv_cell_size)
        .method("prob_lo", &JlGeometry::prob_lo)
        .method("prob_hi", &JlGeometry::prob_hi)
        .method("domain", &JlGeometry::domain)
        .method("prob_domain", &JlGeometry::prob_domain)
        .method("is_periodic", &JlGeometry::is_periodic);

    // JlBoxArray
    mod.add_type<JlBoxArray>("JlBoxArray")
        .constructor<>()
        .constructor<const JlBox&>()
        .method("max_size!", &JlBoxArray::max_size)
        .method("convert_to_nodal", &JlBoxArray::convert_to_nodal)
        .method("size", &JlBoxArray::size)
        .method("empty", &JlBoxArray::empty)
        .method("get_box", &JlBoxArray::get_box)
        .method("contains_cell", &JlBoxArray::contains_cell);

    // JlDistributionMapping
    mod.add_type<JlDistributionMapping>("JlDistributionMapping")
        .constructor<>()
        .constructor<const JlBoxArray&>()
        .method("size", &JlDistributionMapping::size)
        .method("get_rank", &JlDistributionMapping::get_rank);

    // JlMultiFab
    mod.add_type<JlMultiFab>("JlMultiFab")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int, int, int>()
        .method("n_comp", &JlMultiFab::n_comp)
        .method("n_grow", &JlMultiFab::n_grow)
        .method("size", &JlMultiFab::size)
        .method("local_size", &JlMultiFab::local_size)
        .method("box_array", &JlMultiFab::box_array)
        .method("dmap", &JlMultiFab::dmap)
        .method("set_val!", &JlMultiFab::set_val)
        .method("set_val_comp!", &JlMultiFab::set_val_comp)
        .method("plus_val!", &JlMultiFab::plus_val)
        .method("mult_val!", &JlMultiFab::mult_val)
        .method("add_from!", &JlMultiFab::add_from)
        .method("copy_from!", &JlMultiFab::copy_from)
        .method("parallel_copy_from!", &JlMultiFab::parallel_copy_from)
        .method("fill_boundary!", &JlMultiFab::fill_boundary)
        .method("fill_boundary_comp!", &JlMultiFab::fill_boundary_comp)
        .method("sum_boundary!", &JlMultiFab::sum_boundary)
        .method("sum_boundary_comp!", &JlMultiFab::sum_boundary_comp)
        .method("min_val", &JlMultiFab::min_val)
        .method("max_val", &JlMultiFab::max_val)
        .method("norm0", &JlMultiFab::norm0)
        .method("norm1", &JlMultiFab::norm1)
        .method("norm2", &JlMultiFab::norm2)
        .method("sum_val", &JlMultiFab::sum_val);

    // JliMultiFab
    mod.add_type<JliMultiFab>("JliMultiFab")
        .constructor<>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int>()
        .constructor<const JlBoxArray&, const JlDistributionMapping&, int, int, int, int>()
        .method("n_comp", &JliMultiFab::n_comp)
        .method("n_grow", &JliMultiFab::n_grow)
        .method("size", &JliMultiFab::size)
        .method("local_size", &JliMultiFab::local_size)
        .method("box_array", &JliMultiFab::box_array)
        .method("dmap", &JliMultiFab::dmap)
        .method("set_val!", &JliMultiFab::set_val)
        .method("copy_from!", &JliMultiFab::copy_from)
        .method("fill_boundary!", &JliMultiFab::fill_boundary)
        .method("sum_boundary!", &JliMultiFab::sum_boundary)
        .method("min_val", &JliMultiFab::min_val)
        .method("max_val", &JliMultiFab::max_val);

    // JlMFIter
    mod.add_type<JlMFIter>("JlMFIter")
        .constructor<const JlMultiFab&, bool>()
        .constructor<const JliMultiFab&, bool>()
        .method("is_valid", &JlMFIter::is_valid)
        .method("next!", &JlMFIter::next)
        .method("close!", &JlMFIter::close)
        .method("index", &JlMFIter::index)
        .method("local_tile_index", &JlMFIter::local_tile_index)
        .method("valid_box", &JlMFIter::valid_box)
        .method("tile_box", &JlMFIter::tile_box)
        .method("fab_box", &JlMFIter::fab_box)
        .method("fab_lo", &JlMFIter::fab_lo)
        .method("fab_hi", &JlMFIter::fab_hi)
        .method("ifab_lo", &JlMFIter::ifab_lo)
        .method("ifab_hi", &JlMFIter::ifab_hi)
        .method("data_ptr", &JlMFIter::data_ptr)
        .method("idata_ptr", &JlMFIter::idata_ptr);

    // JlAmrCore
    mod.add_type<JlAmrCore>("JlAmrCore")
        .constructor<const JlRealBox&, int, int, int, int, int,
                     bool, bool, bool, int, int, int, int, int, int, double>()
        .method("add_refine_box!", &JlAmrCore::add_refine_box)
        .method("clear_refine_boxes!", &JlAmrCore::clear_refine_boxes)
        .method("init_from_scratch!", &JlAmrCore::init_from_scratch)
        .method("regrid_levels!", &JlAmrCore::regrid_levels)
        .method("finest_level", &JlAmrCore::get_finest_level)
        .method("max_level", &JlAmrCore::get_max_level)
        .method("geom", &JlAmrCore::get_geom)
        .method("box_array", &JlAmrCore::get_box_array)
        .method("node_box_array", &JlAmrCore::get_node_box_array)
        .method("dmap", &JlAmrCore::get_dmap)
        .method("ref_ratio", &JlAmrCore::get_ref_ratio)
        .method("set_box_array!", &JlAmrCore::set_box_array)
        .method("set_dmap!", &JlAmrCore::set_dmap)
        .method("set_finest_level!", &JlAmrCore::set_finest_level);

    // Coarse-Fine Interpolation & Multi-level Utilities
    mod.method("interp_from_coarse_to_fine!", [](JlMultiFab& coarse, JlMultiFab& fine,
                                                 int icomp, int ncomp,
                                                 int rx, int ry, int rz,
                                                 const JlGeometry& cgeom,
                                                 const JlGeometry& fgeom,
                                                 int interp_type, int nghost) {
        amrex::MultiFab f(*fine.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab c(*coarse.mf, amrex::make_alias, icomp, ncomp);
        int lo_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        int hi_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        amrex::Vector<amrex::BCRec> bcs(ncomp, amrex::BCRec(lo_bc, hi_bc));
        PhysBCFunctNoOpFab<amrex::FArrayBox> cphysbc, fphysbc;
        amrex::Interpolater* mapper = get_interpolater(interp_type);
        amrex::InterpFromCoarseLevel(f, amrex::IntVect(nghost), 0.0, c, 0, 0, ncomp,
                                     cgeom.geom, fgeom.geom, cphysbc, 0, fphysbc, 0,
                                     amrex::IntVect(rx, ry, rz), mapper, bcs, 0);
    });

    mod.method("fill_patch_single_level!", [](JlMultiFab& dst, double time,
                                              JlMultiFab& src, int scomp, int dcomp, int ncomp,
                                              const JlGeometry& geom) {
        PhysBCFunctNoOpFab<amrex::FArrayBox> physbc;
        amrex::Vector<amrex::MultiFab*> smf{src.mf.get()};
        amrex::Vector<amrex::Real> stime{time};
        amrex::FillPatchSingleLevel(*dst.mf, time, smf, stime, scomp, dcomp, ncomp,
                                    geom.geom, physbc, 0);
    });

    mod.method("fill_patch_two_levels!", [](JlMultiFab& dst, double time,
                                            JlMultiFab& cmf, JlMultiFab& fmf,
                                            int scomp, int dcomp, int ncomp,
                                            const JlGeometry& cgeom, const JlGeometry& fgeom,
                                            int rx, int ry, int rz, int interp_type) {
        PhysBCFunctNoOpFab<amrex::FArrayBox> cphysbc, fphysbc;
        amrex::Vector<amrex::MultiFab*> cmf_vec{cmf.mf.get()};
        amrex::Vector<amrex::MultiFab*> fmf_vec{fmf.mf.get()};
        amrex::Vector<amrex::Real> ct{time}, ft{time};
        int lo_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        int hi_bc[3] = {amrex::BCType::int_dir, amrex::BCType::int_dir, amrex::BCType::int_dir};
        amrex::Vector<amrex::BCRec> bcs(ncomp, amrex::BCRec(lo_bc, hi_bc));
        amrex::Interpolater* mapper = get_interpolater(interp_type);
        amrex::FillPatchTwoLevels(*dst.mf, time, cmf_vec, ct, fmf_vec, ft,
                                  scomp, dcomp, ncomp, cgeom.geom, fgeom.geom,
                                  cphysbc, 0, fphysbc, 0,
                                  amrex::IntVect(rx, ry, rz), mapper, bcs, 0);
    });

    mod.method("average_down!", [](const JlMultiFab& fine, JlMultiFab& coarse,
                                   const JlGeometry& fgeom, const JlGeometry& cgeom,
                                   int scomp, int ncomp, int rx, int ry, int rz) {
        amrex::average_down(*fine.mf, *coarse.mf, fgeom.geom, cgeom.geom,
                            scomp, ncomp, amrex::IntVect(rx, ry, rz));
    });

    mod.method("average_down_nodal!", [](const JlMultiFab& fine, JlMultiFab& coarse,
                                         int rx, int ry, int rz, int nghost) {
        amrex::average_down_nodal(*fine.mf, *coarse.mf,
                                  amrex::IntVect(rx, ry, rz), nghost);
    });

    mod.method("sum_fine_to_coarse_nodal!", [](JlMultiFab& coarse, const JlMultiFab& fine,
                                               int icomp, int ncomp,
                                               int rx, int ry, int rz) {
        amrex::MultiFab f(*fine.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab c(*coarse.mf, amrex::make_alias, icomp, ncomp);
        amrex::MultiFab ctmp(c.boxArray(), c.DistributionMap(), ncomp, 0);
        ctmp.setVal(0.0);
        amrex::average_down_nodal(f, ctmp, amrex::IntVect(rx, ry, rz));
        amrex::MultiFab::Add(c, ctmp, 0, 0, ncomp, 0);
    });

    // JlParticleContainer & JlParIter
    mod.add_type<JlParticleContainer>("JlParticleContainer")
        .constructor<JlAmrCore&, int, int>()
        .constructor<const JlGeometry&, const JlDistributionMapping&, const JlBoxArray&, int, int>()
        .method("reserve_and_resize_data!", &JlParticleContainer::reserve_and_resize_data)
        .method("ensure_tile!", &JlParticleContainer::ensure_tile)
        .method("resize_tile!", &JlParticleContainer::resize_tile)
        .method("tile_num_particles", &JlParticleContainer::tile_num_particles)
        .method("tile_idcpu_ptr", &JlParticleContainer::tile_idcpu_ptr)
        .method("tile_pos_ptr", &JlParticleContainer::tile_pos_ptr)
        .method("tile_real_ptr", &JlParticleContainer::tile_real_ptr)
        .method("tile_int_ptr", &JlParticleContainer::tile_int_ptr)
        .method("redistribute!", &JlParticleContainer::redistribute)
        .method("total_number_of_particles", &JlParticleContainer::total_number_of_particles)
        .method("number_of_particles_at_level", &JlParticleContainer::number_of_particles_at_level)
        .method("clear_particles!", &JlParticleContainer::clear_particles)
        .method("write_plotfile", &JlParticleContainer::write_plotfile)
        .method("checkpoint", &JlParticleContainer::checkpoint);

    mod.add_type<JlParIter>("JlParIter")
        .constructor<JlParticleContainer&, int>()
        .method("is_valid", &JlParIter::is_valid)
        .method("next!", &JlParIter::next)
        .method("close!", &JlParIter::close)
        .method("index", &JlParIter::index)
        .method("local_tile_index", &JlParIter::local_tile_index)
        .method("valid_box", &JlParIter::valid_box)
        .method("tile_box", &JlParIter::tile_box)
        .method("num_particles", &JlParIter::num_particles)
        .method("resize_tile!", &JlParIter::resize_tile)
        .method("idcpu_ptr", &JlParIter::idcpu_ptr)
        .method("pos_ptr", &JlParIter::pos_ptr)
        .method("real_ptr", &JlParIter::real_ptr)
        .method("int_ptr", &JlParIter::int_ptr);

    mod.method("next_particle_id", []() {
        return static_cast<int64_t>(amrex::Particle<0, 0>::NextID());
    });

    // Plotfile I/O
    mod.method("write_single_level_plotfile", [](const std::string& plotfilename,
                                                 const JlMultiFab& mf,
                                                 const std::string& varnames_csv,
                                                 const JlGeometry& geom,
                                                 double time, int level_step) {
        auto vnames = split_varnames(varnames_csv, mf.n_comp());
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::WriteSingleLevelPlotfile(plotfilename, *mf.mf, amrex_vnames,
                                        geom.geom, time, level_step);
    });

    mod.method("write_two_level_plotfile", [](const std::string& plotfilename,
                                              const JlMultiFab& mf0,
                                              const JlMultiFab& mf1,
                                              const std::string& varnames_csv,
                                              const JlGeometry& geom0,
                                              const JlGeometry& geom1,
                                              double time, int step,
                                              int rx, int ry, int rz) {
        auto vnames = split_varnames(varnames_csv, mf0.n_comp());
        amrex::Vector<std::string> amrex_vnames(vnames.begin(), vnames.end());
        amrex::Vector<const amrex::MultiFab*> mfs{mf0.mf.get(), mf1.mf.get()};
        amrex::Vector<amrex::Geometry> geoms{geom0.geom, geom1.geom};
        amrex::Vector<int> steps{step, step};
        amrex::Vector<amrex::IntVect> rr{amrex::IntVect(rx, ry, rz)};
        amrex::WriteMultiLevelPlotfile(plotfilename, 2, mfs, amrex_vnames,
                                       geoms, time, steps, rr);
    });
}
