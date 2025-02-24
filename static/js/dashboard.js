import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getDatabase, ref, get, update } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-database.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";
import { getStorage, ref as storageRef, uploadBytes, getDownloadURL } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-storage.js";

// Firebase initialization
const firebaseConfig = window.firebaseConfig;
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);
const storage = getStorage(app);

// Listen for auth state changes
onAuthStateChanged(auth, (user) => {
    if (user) {
        loadProfileData(user.uid);
    } else {
        console.log("User not logged in.");
    }
});

// Fetch data from both "lawyers" and "lawyers_profile" tables
function loadProfileData(userId) {
    const lawyerRef = ref(database, `lawyers/${userId}`);
    const profileRef = ref(database, `lawyers_profile/${userId}`);

    // Fetch from "lawyers" table (Basic Info)
    get(lawyerRef).then((snapshot) => {
        if (snapshot.exists()) {
            const lawyerData = snapshot.val();
            console.log("Lawyer Data:", lawyerData);

            document.getElementById("lawyerName").textContent = lawyerData.name || "Not Provided";
            document.getElementById("lawyerEmail").textContent = lawyerData.email || "Not Provided";
            document.getElementById("barCouncilID").textContent = lawyerData.barCouncilID || "Not Provided";
        } else {
            console.log("No data found in 'lawyers' table.");
        }
    }).catch(error => console.error("Error fetching 'lawyers' data:", error));

    // Fetch from "lawyers_profile" table (Profile Details)
    get(profileRef).then((snapshot) => {
        if (snapshot.exists()) {
            const profileData = snapshot.val();
            console.log("Profile Data:", profileData);

            document.getElementById("displayName").textContent = profileData.name ;
            document.getElementById("displayContact").textContent = profileData.contact ;
            document.getElementById("displayAddress").textContent = profileData.address ;
            document.getElementById("displayFirmName").textContent = profileData.firmName ;
            document.getElementById("displayFirmSize").textContent = profileData.firmSize ;
            document.getElementById("displayAffiliation").textContent = profileData.affiliation ;
            document.getElementById("displayPracticeArea").textContent = profileData.practiceArea ;
            document.getElementById("displayDesignation").textContent = profileData.designation ;
            document.getElementById("displayExperience").textContent = profileData.experience ;
            document.getElementById("displayCases").textContent = profileData.cases ;
            document.getElementById("displayFees").textContent = profileData.fees ;

            if (profileData.profilePicUrl) {
                document.getElementById("profilePic").src = profileData.profilePicUrl;
            }
        } else {
            console.log("No data found in 'lawyers_profile' table.");
        }
    }).catch(error => console.error("Error fetching profile data:", error));
}

// Open the Edit Profile modal
window.openEditProfile = function () {
    document.getElementById("editProfileModal").style.display = "flex";
    loadProfileIntoForm();
};

// Close the Edit Profile modal
window.closeEditProfile = function () {
    document.getElementById("editProfileModal").style.display = "none";
};

// Load current profile data into the form
function loadProfileIntoForm() {
    document.getElementById("name").value = document.getElementById("lawyerName").textContent;
    document.getElementById("email").value = document.getElementById("lawyerEmail").textContent;
    document.getElementById("contact").value = document.getElementById("displayContact").textContent;
    document.getElementById("address").value = document.getElementById("displayAddress").textContent;
    document.getElementById("firmName").value = document.getElementById("displayFirmName").textContent;
    document.getElementById("firmSize").value = document.getElementById("displayFirmSize").textContent;
    document.getElementById("affiliation").value = document.getElementById("displayAffiliation").textContent;
    document.getElementById("practiceArea").value = document.getElementById("displayPracticeArea").textContent;
    document.getElementById("designation").value = document.getElementById("displayDesignation").textContent;
    document.getElementById("experience").value = document.getElementById("displayExperience").textContent;
    document.getElementById("cases").value = document.getElementById("displayCases").textContent;
    document.getElementById("fees").value = document.getElementById("displayFees").textContent;
}

// Handle Edit Profile form submission
document.getElementById("editProfileForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const user = auth.currentUser;
    if (!user) {
        alert("User not authenticated!");
        return;
    }

    const updatedData = {
        name: document.getElementById("name").value,
        contact: document.getElementById("contact").value,
        address: document.getElementById("address").value,
        firmName: document.getElementById("firmName").value,
        firmSize: document.getElementById("firmSize").value,
        affiliation: document.getElementById("affiliation").value,
        practiceArea: document.getElementById("practiceArea").value,
        designation: document.getElementById("designation").value,
        experience: document.getElementById("experience").value,
        cases: document.getElementById("cases").value,
        fees: document.getElementById("fees").value
    };

    const profileRef = ref(database, `lawyers_profile/${user.uid}`);
    update(profileRef, updatedData)
        .then(() => {
            alert("Profile updated successfully!");
            closeEditProfile();
            loadProfileData(user.uid); // Refresh displayed data
        })
        .catch(error => console.error("Error updating profile:", error));
});

// Upload and update profile picture
document.getElementById("profilePic").addEventListener("click", function () {
    document.getElementById("profilePicInput").click();
});

document.getElementById("profilePicInput").addEventListener("change", function (event) {
    const file = event.target.files[0];
    if (!file) return;

    const user = auth.currentUser;
    if (!user) {
        alert("User not authenticated!");
        return;
    }

    const storageReference = storageRef(storage, `lawyers_profile/${user.uid}/profilePic.jpg`);
    uploadBytes(storageReference, file)
        .then(() => getDownloadURL(storageReference))
        .then(url => {
            document.getElementById("profilePic").src = url;
            return update(ref(database, `lawyers_profile/${user.uid}`), { profilePicUrl: url });
        })
        .then(() => alert("Profile picture updated successfully!"))
        .catch(error => console.error("Error updating profile picture:", error));
});